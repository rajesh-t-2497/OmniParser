from typing import Optional, List, Dict
import base64
import io
from pydantic import BaseModel
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np

from util.utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img

# Initialize models
yolo_model = get_yolo_model(model_path='weights/icon_detect/model.pt')
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence")
DEVICE = torch.device('cuda')

app = FastAPI(title="OmniParser API", description="API for OmniParser screen parsing tool")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class OmniParserResponse(BaseModel):
    image_base64: str
    parsed_elements: List[str]

@app.post("/process", response_model=OmniParserResponse)
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = Form(0.05),
    iou_threshold: float = Form(0.1),
    use_paddleocr: bool = Form(True),
    imgsz: int = Form(640)
):
    """
    Process an image with OmniParser to detect and parse UI elements.
    
    - **file**: The image file to process
    - **box_threshold**: Threshold for removing bounding boxes with low confidence (0.01-1.0)
    - **iou_threshold**: Threshold for removing bounding boxes with large overlap (0.01-1.0)
    - **use_paddleocr**: Whether to use PaddleOCR for text detection
    - **imgsz**: Image size for icon detection (640-1920)
    
    Returns processed image and parsed screen elements.
    """
    try:
        # Read the image file
        contents = await file.read()
        image_input = Image.open(io.BytesIO(contents))
        
        # Process the image
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        # OCR processing
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Get labeled image
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input, 
            yolo_model, 
            BOX_TRESHOLD=box_threshold, 
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=caption_model_processor, 
            ocr_text=text,
            iou_threshold=iou_threshold, 
            imgsz=imgsz
        )
        
        # Format the response
        parsed_elements = [f"icon {i}: {str(v)}" for i, v in enumerate(parsed_content_list)]
        
        return {
            "image_base64": dino_labled_img,  # Already base64 encoded from get_som_labeled_img
            "parsed_elements": parsed_elements
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Response model for OCR-only API
class OCRResponse(BaseModel):
    text: List[str]
    boxes: List[Dict]
    image_base64: Optional[str] = None

@app.post("/ocr", response_model=OCRResponse)
async def ocr_only(
    file: UploadFile = File(...),
    use_paddleocr: bool = Form(True)
):
    try:
        # Debugging log
        print("Received OCR request")
        
        # Read the image file
        contents = await file.read()
        print("File read successfully")
        
        image_input = Image.open(io.BytesIO(contents))
        print("Image opened successfully")
        
        # OCR processing
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        print("OCR processing completed")
        
        text, ocr_bbox = ocr_bbox_rslt
        
        # Use get_som_labeled_img to generate the annotated image
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }
        
        dino_labeled_img, _, _ = get_som_labeled_img(
            image_input, 
            None,  # No YOLO model needed for OCR-only
            BOX_TRESHOLD=0.0,  # Not applicable for OCR-only
            output_coord_in_ratio=True, 
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config, 
            caption_model_processor=None,  # No captioning needed for OCR-only
            ocr_text=text,
            iou_threshold=0.0,  # Not applicable for OCR-only
            imgsz=640  # Default image size
        )
        
        # Convert the annotated image to base64
        img_base64 = dino_labeled_img  # Already base64 encoded by get_som_labeled_img
        
        print("Response prepared successfully")
        return {
            "text": text,
            "boxes": [
                {
                    "id": i,
                    "text": text[i],
                    "box": box.tolist() if hasattr(box, 'tolist') else box,
                    "confidence": box[4] if len(box) > 4 else None
                }
                for i, box in enumerate(ocr_bbox)
            ],
            "image_base64": img_base64
        }
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    #uvicorn.run(app, host="0.0.0.0", port=8000)
    uvicorn.run("omni_parser_api:app", host="0.0.0.0", port=8000, reload=True, workers=1)
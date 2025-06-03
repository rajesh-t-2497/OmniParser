from typing import List, Dict
import io
import base64
from pydantic import BaseModel
import torch
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2

from util.utils import check_ocr_box

app = FastAPI(title="OCR API", description="API for text extraction from images")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Response model
class OCRResponse(BaseModel):
    text: List[str]
    boxes: List[Dict]
    image_base64: str

def draw_ocr_boxes(image, boxes, texts):
    """Draw OCR bounding boxes on the image"""
    img_np = np.array(image)
    
    # Convert to BGR if needed (for OpenCV)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    box_overlay_ratio = image.size[0] / 3200
    text_scale = 0.8 * box_overlay_ratio
    text_thickness = max(int(2 * box_overlay_ratio), 1)
    box_thickness = max(int(3 * box_overlay_ratio), 1)
    
    # Draw boxes and text
    for i, (box, text) in enumerate(zip(boxes, texts)):
        if hasattr(box, 'tolist'):
            box = box.tolist()
        
        # Extract coordinates (assuming xyxy format)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # Draw rectangle
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        
        # Add text
        cv2.putText(img_np, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    text_scale, (0, 255, 0), text_thickness)
    
    # Convert back to RGB for PIL
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_np)

@app.post("/ocr", response_model=OCRResponse)
async def extract_text(
    file: UploadFile = File(...),
    use_paddleocr: bool = Form(True)
):
    """
    Extract text from an image using OCR.
    
    - **file**: The image file to process
    - **use_paddleocr**: Whether to use PaddleOCR (True) or EasyOCR (False)
    
    Returns extracted text, bounding boxes, and a base64 encoded image with annotations.
    """
    try:
        # Read the image file
        contents = await file.read()
        image_input = Image.open(io.BytesIO(contents))
        
        # OCR processing
        ocr_bbox_rslt, _ = check_ocr_box(
            image_input, 
            display_img=False, 
            output_bb_format='xyxy', 
            goal_filtering=None, 
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}, 
            use_paddleocr=use_paddleocr
        )
        text, ocr_bbox = ocr_bbox_rslt
        
        # Convert OCR bounding boxes to a more API-friendly format
        formatted_boxes = []
        for i, box in enumerate(ocr_bbox):
            formatted_boxes.append({
                "id": i,
                "text": text[i],
                "box": box.tolist() if hasattr(box, 'tolist') else box,
                "confidence": box[4] if len(box) > 4 else None
            })
        
        # Create an image with OCR annotations for visualization
        annotated_image = draw_ocr_boxes(image_input, ocr_bbox, text)
        
        # Convert the annotated image to base64
        buffered = io.BytesIO()
        annotated_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            "text": text,
            "boxes": formatted_boxes,
            "image_base64": img_base64
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "OCR API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ocr_only_api:app", host="0.0.0.0", port=8002, reload=True, workers=1)
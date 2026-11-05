import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, File, UploadFile, Form
from app.models.api_models import (
    ExamGenerationRequest, BatchExamGenerationRequest, QuestionGenerationRequest, QuestionRegenerationRequest,
    AssignmentGenerationRequest, CurriculumIngestRequest, APIResponse
)
from app.models.curriculum import BoardType
from app.curriculum.board_profiles import BoardProfileRegistry
from app.curriculum.loader import CurriculumLoader
from app.ocr.service import OCRService
from app.pipeline import AcademicContentPipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["AI Academic Engine"])
pipeline = AcademicContentPipeline()

@router.get("/health", response_model=APIResponse)
def health_check():
    return APIResponse(
        success=True,
        message="AI Academic Content Engine is operational.",
        data={"status": "UP", "version": "1.0.0"}
    )

@router.get("/board-profiles", response_model=APIResponse)
def get_supported_board_profiles():
    try:
        profiles = [
            BoardProfileRegistry.get_profile(BoardType.CBSE, "2024-25"),
            BoardProfileRegistry.get_profile(BoardType.CBSE, "2025-26"),
            BoardProfileRegistry.get_profile(BoardType.ICSE, "2024-25"),
            BoardProfileRegistry.get_profile(BoardType.UP_BOARD, "2024-25")
        ]
        return APIResponse(
            success=True,
            message="Retrieved supported board profiles.",
            data=[p.model_dump() for p in profiles]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/exam/generate", response_model=APIResponse)
def generate_exam_paper(request: ExamGenerationRequest):
    try:
        paper, quality, warnings = pipeline.generate_validated_exam_paper(request)
        return APIResponse(
            success=True,
            message="Exam paper successfully generated.",
            data={
                "exam_paper": paper.model_dump()
            }
        )
    except Exception as e:
        logger.error(f"Error generating exam paper: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate exam paper: {str(e)}")

@router.post("/exam/batch-generate", response_model=APIResponse)
def batch_generate_exam_papers(batch_request: BatchExamGenerationRequest):
    """Generates multiple exam papers concurrently in parallel (e.g. Class 1 to 10 across subjects)
    with zero redundant cold-start delays or duplicate chunking overhead.
    """
    try:
        results = pipeline.generate_batch_exam_papers(
            requests=batch_request.requests,
            max_workers=batch_request.max_workers
        )
        batch_data = []
        for paper, quality, warnings in results:
            batch_data.append({
                "exam_paper": paper.model_dump(),
                "quality_evaluation": quality.model_dump(),
                "warnings": warnings
            })
        return APIResponse(
            success=True,
            message=f"Successfully generated {len(results)} exam papers in batch.",
            data={"generated_papers": batch_data}
        )
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch paper generation failed: {str(e)}")

@router.post("/question/generate", response_model=APIResponse)
def generate_single_question(request: QuestionGenerationRequest):
    try:
        from app.models.blueprint import QuestionBlueprint
        bp = QuestionBlueprint(
            id=f"Q_GEN_{request.chapter_id}_{request.topic_id}",
            question_type=request.question_type,
            marks=request.marks,
            difficulty=request.difficulty,
            chapter_id=request.chapter_id,
            topic_id=request.topic_id
        )
        q = pipeline.question_generator.generate_question(
            bp=bp,
            board=request.board,
            academic_year=request.academic_year,
            class_name=request.class_name,
            subject=request.subject,
            custom_prompt=request.prompt
        )
        return APIResponse(
            success=True,
            message="Question generated successfully.",
            data=q.model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/question/regenerate", response_model=APIResponse)
def regenerate_question(request: QuestionRegenerationRequest):
    try:
        new_q = pipeline.question_generator.regenerate_targeted_question(
            bp=request.blueprint,
            board=request.board,
            academic_year=request.academic_year,
            class_name=request.class_name,
            subject=request.subject,
            failed_question_stem="Previous failed question",
            rejection_reason=request.rejection_reason
        )
        return APIResponse(
            success=True,
            message="Question regenerated successfully.",
            data=new_q.model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/assignment/generate", response_model=APIResponse)
def generate_assignment(request: AssignmentGenerationRequest):
    try:
        assignment = pipeline.assignment_generator.generate_assignment(
            board=request.board,
            academic_year=request.academic_year,
            class_name=request.class_name,
            subject=request.subject,
            total_marks=request.total_marks,
            title=request.title,
            chapter_ids=request.chapter_ids
        )
        return APIResponse(
            success=True,
            message="Assignment generated successfully.",
            data=assignment.model_dump()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/curriculum/index", response_model=APIResponse)
def index_curriculum(request: CurriculumIngestRequest):
    try:
        curriculum = CurriculumLoader.parse_raw_text(
            raw_text=request.content,
            board=request.board,
            academic_year=request.academic_year,
            class_name=request.class_name,
            subject=request.subject
        )
        chunks = CurriculumChunker.chunk_curriculum(curriculum)
        raw_chunks = [{"text": c.text, "metadata": c.metadata} for c in chunks]
        pipeline.rag_retriever.index_curriculum_chunks(raw_chunks)
        
        return APIResponse(
            success=True,
            message=f"Indexed {len(chunks)} curriculum chunks for {request.board} {request.class_name} {request.subject}.",
            data={"indexed_chunks_count": len(chunks)}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/exam/generate-from-image", response_model=APIResponse)
async def generate_exam_paper_from_image(
    file: Optional[UploadFile] = File(None, description="Uploaded image (textbook page, handwritten note, diagram)"),
    prompt: str = Form("10 mcq based on this image", description="Prompt specifying question types and count e.g. '10 mcq' or '5 short questions'")
):
    """Uploads an image, extracts text via Python OCR, and generates an exam paper based strictly on the user's prompt specifying question types and count."""
    try:
        ocr_text = ""
        filename = "image"
        if file and file.filename:
            filename = file.filename
            contents = await file.read()
            if contents:
                ocr_text = OCRService.extract_text_from_bytes(contents, filename=filename)
                logger.info(f"Extracted {len(ocr_text)} characters via Python OCR from uploaded image: {filename}")

        combined_prompt = f"{prompt}\n\n=== Extracted Image Text (via Python OCR) ===\n{ocr_text}" if ocr_text else prompt

        gen_request = ExamGenerationRequest(
            prompt=combined_prompt,
            title=f"Test from Image ({filename})"
        )

        paper, quality, warnings = pipeline.generate_validated_exam_paper(gen_request)

        return APIResponse(
            success=True,
            message="Exam paper successfully generated from image.",
            data={
                "exam_paper": paper.model_dump()
            }
        )
    except Exception as e:
        logger.error(f"Failed to generate exam paper from image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image-to-Test generation failed: {str(e)}")


import math
import re
import logging
from typing import List, Dict, Optional, Tuple
from app.models.curriculum import SubjectCurriculum, BoardType, DifficultyLevel, QuestionType
from app.models.blueprint import ExamBlueprint, SectionBlueprint, QuestionBlueprint
from app.curriculum.board_profiles import BoardProfileRegistry

logger = logging.getLogger(__name__)

class BlueprintEngine:
    """Blueprint Engine generates explicit, deterministic exam blueprints.
    
    The LLM must NEVER freely decide total marks, question count, or section distribution.
    All parameters are strictly controlled by the Blueprint Engine based on Board profiles
    and curriculum topic weightages.
    """

    @classmethod
    def generate_blueprint(
        cls,
        curriculum: SubjectCurriculum,
        total_marks: float = 80.0,
        duration_minutes: int = 180,
        title: Optional[str] = None,
        custom_difficulty: Optional[Dict[DifficultyLevel, float]] = None,
        prompt: Optional[str] = None
    ) -> ExamBlueprint:
        profile = BoardProfileRegistry.get_profile(curriculum.board, curriculum.academic_year)

        board_val = curriculum.board.value if hasattr(curriculum.board, "value") else str(curriculum.board)
        exam_title = title or f"{board_val} {curriculum.class_name} {curriculum.subject} Examination ({curriculum.academic_year})"
        
        # Difficulty distribution default: 30% Easy, 50% Medium, 20% Hard
        diff_dist = custom_difficulty or {
            DifficultyLevel.EASY: 0.30,
            DifficultyLevel.MEDIUM: 0.50,
            DifficultyLevel.HARD: 0.20
        }

        # Gather topics from curriculum
        all_topics: List[Tuple[str, str]] = []  # (chapter_id, topic_id)
        for chapter in curriculum.chapters:
            for topic in chapter.topics:
                all_topics.append((chapter.id, topic.id))

        if not all_topics:
            all_topics = [("CH_01", "CH_01_TOPIC_01")]

        p_lower = (prompt or "").lower()

        # 1. Check if user specified a specific chapter in prompt (e.g. "chapter 1", "ch 1", "chapter q" typo for 1)
        ch_match = re.search(r"(?:chapter|ch)\s*([0-9]+|[a-zA-Z])", p_lower)
        if ch_match:
            raw_ch = ch_match.group(1).lower()
            ch_num = int(raw_ch) if raw_ch.isdigit() else (1 if raw_ch == "q" else None)
            if ch_num:
                matching_chapters = [
                    c for idx, c in enumerate(curriculum.chapters, 1)
                    if idx == ch_num or c.id.endswith(f"_{ch_num:02d}") or c.id.endswith(f"_{ch_num}") or f"chapter {ch_num}" in c.name.lower()
                ]
                if matching_chapters:
                    filtered_topics = []
                    for ch in matching_chapters:
                        for topic in ch.topics:
                            filtered_topics.append((ch.id, topic.id))
                    if filtered_topics:
                        all_topics = filtered_topics
                        logger.info(f"Filtered blueprint topics to Chapter {ch_num}: {len(all_topics)} topics found.")

        # 2. Check if user specified exact question types and counts in prompt
        # (e.g. "10 mcq", "5 short answer", "2 long answer", "3 case study", "10 questions")
        specs_found = []

        mcq_m = re.search(r"(\d+)\s*(?:mcqs?|multiple\s*choice|objective)", p_lower)
        if mcq_m:
            count = int(mcq_m.group(1))
            specs_found.append({
                "name": "Section A (Multiple Choice Questions)",
                "type": QuestionType.MCQ,
                "default_marks": 1.0,
                "instruction": "Select the correct option for each question.",
                "count": count
            })

        sa_m = re.search(r"(\d+)\s*(?:short\s*(?:answer)?(?:\s*questions?)?|sa\b)", p_lower)
        if sa_m:
            count = int(sa_m.group(1))
            sec_letter = chr(ord('A') + len(specs_found))
            specs_found.append({
                "name": f"Section {sec_letter} (Short Answer Questions)",
                "type": QuestionType.SHORT_ANSWER,
                "default_marks": 2.0 if count > 5 else 3.0,
                "instruction": "Answer the following short answer questions concisely.",
                "count": count
            })

        la_m = re.search(r"(\d+)\s*(?:long\s*(?:answer)?(?:\s*questions?)?|la\b)", p_lower)
        if la_m:
            count = int(la_m.group(1))
            sec_letter = chr(ord('A') + len(specs_found))
            specs_found.append({
                "name": f"Section {sec_letter} (Long Answer Questions)",
                "type": QuestionType.LONG_ANSWER,
                "default_marks": 5.0,
                "instruction": "Answer the following long answer questions with detailed explanations.",
                "count": count
            })

        cs_m = re.search(r"(\d+)\s*(?:case\s*(?:based|study)?(?:\s*questions?)?)", p_lower)
        if cs_m:
            count = int(cs_m.group(1))
            sec_letter = chr(ord('A') + len(specs_found))
            specs_found.append({
                "name": f"Section {sec_letter} (Case-Based Questions)",
                "type": QuestionType.CASE_BASED,
                "default_marks": 4.0,
                "instruction": "Read the context and answer the case-based questions.",
                "count": count
            })

        if not specs_found:
            generic_m = re.search(r"(\d+)\s*questions?", p_lower)
            if generic_m:
                count = int(generic_m.group(1))
                specs_found.append({
                    "name": "Section A (Questions)",
                    "type": QuestionType.MCQ,
                    "default_marks": 1.0,
                    "instruction": "Answer all questions.",
                    "count": count
                })
            elif "mcq" in p_lower:
                specs_found.append({
                    "name": "Section A (Multiple Choice Questions)",
                    "type": QuestionType.MCQ,
                    "default_marks": 1.0,
                    "instruction": "Select the correct option for each question.",
                    "count": 10
                })

        if specs_found:
            section_conventions = specs_found
            target_counts = [s["count"] for s in specs_found]
            total_marks = sum(s["count"] * s["default_marks"] for s in specs_found)
            duration_minutes = max(15, int(total_marks * 2.5))
            num_sections = len(section_conventions)
            logger.info(f"Custom prompt-driven blueprint: {num_sections} sections, {sum(target_counts)} questions, {total_marks} marks.")
        else:
            section_conventions = profile.section_conventions
            if not section_conventions:
                section_conventions = [
                    {"name": "Section A", "type": QuestionType.MCQ, "default_marks": 1.0, "instruction": "Multiple Choice Questions."},
                    {"name": "Section B", "type": QuestionType.SHORT_ANSWER, "default_marks": 3.0, "instruction": "Short Answer Questions."},
                    {"name": "Section C", "type": QuestionType.LONG_ANSWER, "default_marks": 5.0, "instruction": "Long Answer Questions."}
                ]
            num_sections = len(section_conventions)

            # Integer-balanced question count allocation:
            if abs(total_marks - 80.0) < 1e-3 and num_sections == 5:
                target_counts = [20, 6, 7, 3, 3]
            elif abs(total_marks - 70.0) < 1e-3 and num_sections == 5:
                target_counts = [15, 5, 6, 3, 3]
            elif abs(total_marks - 50.0) < 1e-3 and num_sections == 5:
                target_counts = [12, 4, 4, 2, 2]
            else:
                target_counts = []
                for i, sec in enumerate(section_conventions):
                    dm = sec.get("default_marks", 1.0)
                    if num_sections == 5:
                        share_pct = [0.25, 0.15, 0.26, 0.19, 0.15][i]
                        target_marks = total_marks * share_pct
                    else:
                        target_marks = total_marks / num_sections
                    cnt = max(1, round(target_marks / dm))
                    target_counts.append(cnt)

            current_sum = sum(target_counts[i] * section_conventions[i].get("default_marks", 1.0) for i in range(num_sections))
            diff = int(round(total_marks - current_sum))

            sec_1_idx = 0
            for i, sec in enumerate(section_conventions):
                if sec.get("default_marks", 1.0) == 1.0:
                    sec_1_idx = i
                    break

            if diff > 0:
                target_counts[sec_1_idx] += diff
            elif diff < 0:
                to_remove = abs(diff)
                can_remove_1 = max(0, target_counts[sec_1_idx] - 1)
                rem_from_1 = min(can_remove_1, to_remove)
                target_counts[sec_1_idx] -= rem_from_1
                to_remove -= rem_from_1

                if to_remove > 0:
                    for idx in reversed(range(num_sections)):
                        dm = int(section_conventions[idx].get("default_marks", 1.0))
                        while to_remove >= dm and target_counts[idx] > 1:
                            target_counts[idx] -= 1
                            to_remove -= dm
                        if to_remove <= 0:
                            break

        # Build sections
        difficulty_cycle = [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]
        q_global_counter = 1
        sections: List[SectionBlueprint] = []
        for idx, sec_spec in enumerate(section_conventions):
            sec_name = sec_spec["name"]
            q_type = sec_spec["type"]
            default_marks = sec_spec.get("default_marks", 1.0)
            q_count = max(1, target_counts[idx])
            actual_sec_marks = q_count * default_marks

            question_blueprints: List[QuestionBlueprint] = []
            for q_i in range(q_count):
                ch_id, t_id = all_topics[(q_global_counter - 1) % len(all_topics)]
                diff = difficulty_cycle[(q_global_counter - 1) % len(difficulty_cycle)]

                q_bp = QuestionBlueprint(
                    id=f"Q_{sec_name.replace(' ', '_')}_{q_i + 1}",
                    question_type=q_type,
                    marks=default_marks,
                    difficulty=diff,
                    chapter_id=ch_id,
                    topic_id=t_id,
                    blooms_level="Application" if diff == DifficultyLevel.MEDIUM else ("Knowledge" if diff == DifficultyLevel.EASY else "Analysis")
                )
                question_blueprints.append(q_bp)
                q_global_counter += 1

            sec_bp = SectionBlueprint(
                name=sec_name,
                instructions=sec_spec.get("instruction", "Answer all questions."),
                question_blueprints=question_blueprints,
                total_marks=actual_sec_marks,
                optional_questions_count=1 if profile.allow_internal_choice and actual_sec_marks >= 10 else 0
            )
            sections.append(sec_bp)

        # Final exact total marks audit (handles small totals or fractional targets cleanly)
        allocated_total = sum(sec.total_marks for sec in sections)
        if abs(allocated_total - total_marks) > 1e-3 and allocated_total > 0 and not specs_found:
            ratio = total_marks / allocated_total
            allocated_total = 0.0
            for sec in sections:
                sec_total = 0.0
                for q in sec.question_blueprints:
                    q.marks = max(1.0, round(q.marks * ratio, 1))
                    sec_total += q.marks
                sec.total_marks = sec_total
                allocated_total += sec_total

            diff = round(total_marks - allocated_total, 1)
            if abs(diff) > 1e-3 and sections and sections[-1].question_blueprints:
                sections[-1].question_blueprints[-1].marks = max(1.0, round(sections[-1].question_blueprints[-1].marks + diff, 1))
                sections[-1].total_marks = round(sum(q.marks for q in sections[-1].question_blueprints), 1)
                allocated_total = round(sum(sec.total_marks for sec in sections), 1)

        # Calculate topic distribution map
        topic_marks: Dict[str, float] = {}
        for sec in sections:
            for q in sec.question_blueprints:
                topic_marks[q.topic_id] = topic_marks.get(q.topic_id, 0.0) + q.marks

        topic_dist = {t_id: round(m / allocated_total, 4) for t_id, m in topic_marks.items()} if allocated_total > 0 else {}

        blueprint = ExamBlueprint(
            title=exam_title,
            board=curriculum.board,
            academic_year=curriculum.academic_year,
            class_name=curriculum.class_name,
            subject=curriculum.subject,
            total_marks=allocated_total,
            duration_minutes=duration_minutes,
            sections=sections,
            difficulty_distribution=diff_dist,
            topic_distribution=topic_dist
        )

        logger.info(f"Generated blueprint '{exam_title}' with {len(sections)} sections and {allocated_total} total marks.")
        return blueprint

    @staticmethod
    def _partition_marks(total: float, parts: int) -> List[float]:
        """Partitions total marks into standard ratios for sections."""
        if parts == 1:
            return [total]
        if parts == 2:
            return [round(total * 0.4, 1), round(total * 0.6, 1)]
        if parts == 3:
            return [round(total * 0.3, 1), round(total * 0.4, 1), round(total * 0.3, 1)]
        if parts == 4:
            return [round(total * 0.25, 1), round(total * 0.25, 1), round(total * 0.25, 1), round(total * 0.25, 1)]
        
        # Default even partition
        share = round(total / parts, 1)
        res = [share] * parts
        res[-1] = round(total - sum(res[:-1]), 1)
        return res

from typing import List, Optional

from pydantic import BaseModel, Field, conint


Score20 = conint(ge=0, le=20)


class ScoreBreakdown(BaseModel):
    innovation: Score20 = Field(description="아이디어 혁신성 점수 (0~20)")
    difficulty: Score20 = Field(description="개발 난이도 점수 (0~20)")
    design: Score20 = Field(description="디자인 점수 (0~20)")
    deck_quality: Score20 = Field(description="발표자료 완성도 점수 (0~20)")
    impact: Score20 = Field(description="파급효과 점수 (0~20)")


class ReviewBreakdown(BaseModel):
    innovation_3lines: List[str] = Field(min_length=3, max_length=3)
    difficulty_3lines: List[str] = Field(min_length=3, max_length=3)
    design_3lines: List[str] = Field(min_length=3, max_length=3)
    deck_quality_3lines: List[str] = Field(min_length=3, max_length=3)
    impact_3lines: List[str] = Field(min_length=3, max_length=3)


class JudgeResult(BaseModel):
    scores: ScoreBreakdown
    reviews: ReviewBreakdown
    total_score: Score20 = Field(description="총점(5개 항목 평균, 0~20 정수)")
    final_grade: Optional[str] = Field(default=None, description="S/A/B/C/D 중 하나(없으면 앱에서 계산)")
    overall_review: str


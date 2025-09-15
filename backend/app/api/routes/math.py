from fastapi import APIRouter
from ...models.schemas import MathQuery, MathResponse

router = APIRouter()

@router.post("/solve", response_model=MathResponse)
def solve_math(q: MathQuery):
    # placeholder: route to agents and return steps
    return MathResponse(answer="42", steps=["parse problem", "select tool", "compute"], model="stub")

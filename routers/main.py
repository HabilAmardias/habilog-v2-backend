from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

router = FastAPI()

router.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
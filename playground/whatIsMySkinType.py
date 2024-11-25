from transformers import pipeline
from PIL import Image
import torch  # torch 라이브러리 추가

# 분류 파이프라인 생성, GPU를 사용할 수 있으면 GPU로 설정
pipe = pipeline("image-classification", model="dima806/skin_types_image_detection", device=0 if torch.cuda.is_available() else -1)

# 분석할 이미지 파일 경로 설정
image_path = "source/face.jpg"  # 이미지 상대 경로
try:
    # 이미지 열기 및 전처리
    image = Image.open(image_path).convert("RGB")  # 이미지를 RGB로 변환
    image = image.resize((224, 224))  # 모델에 맞게 크기 조정

    # 피부 상태 분석 수행
    results = pipe(image)

    # 결과 출력
    print("Skin Type Classification Results:")
    for result in results:
        label = result["label"]
        score = result["score"]
        print(f"Label: {label}, Confidence: {score:.2f}")
        image.show()

except FileNotFoundError:
    print(f"Error: File not found at path {image_path}")
except Exception as e:
    print(f"An error occurred: {e}")
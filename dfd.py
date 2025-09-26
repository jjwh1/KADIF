# import pandas as pd
#
# # 기존 submission 불러오기
# submission = pd.read_csv(r"D:\aimers\baseline_submission.csv")
#
# # 첫 번째 열(예: '영업일자')은 그대로 두고, 나머지 수치형 열을 정수(int)로 변환
# for col in submission.columns[1:]:
#     submission[col] = submission[col].fillna(0).astype(int)
#
# # 정수 변환 확인
# print(submission.dtypes)
#
# # 정수형으로 저장
# submission.to_csv(r"D:\aimers\baseline_submission_.csv", index=False, encoding='utf-8')





# import pandas as pd
#
# # 기존 CSV 불러오기
# sub = pd.read_csv(r"D:\aimers\baseline_submission_.csv")  # 기존 파일 경로로 변경
#
# # 다시 저장 (utf-8 인코딩, 소수점 6자리 고정)
# sub.to_csv(r"D:\aimers\baseline_submission__.csv", index=False, encoding="utf-8", float_format="%.6f")
#
# print("저장 완료: final.csv")
#
# import torch
# print(torch.__version__)        # PyTorch 버전
# print(torch.version.cuda)       # PyTorch가 빌드된 CUDA 버전
# print(torch.cuda.is_available())# CUDA 사용 가능 여부
# print(torch.cuda.get_device_name(0)) # 첫 번째 GPU 이름

import os

# 바꿀 폴더 경로 지정
folder = r"C:\Users\8138\Desktop\KADIF\seg\SemanticDataset\labelmap\train\set3"

for fname in os.listdir(folder):
    old_path = os.path.join(folder, fname)
    if os.path.isfile(old_path) and fname.endswith("_gtFine_CategoryId.png"):
        # "000001_leftImg8bit" → "000001_set3_leftImg8bit"
        new_name = fname.replace("_gtFine_CategoryId", "_set3_gtFine_CategoryId")
        new_path = os.path.join(folder, new_name)
        os.rename(old_path, new_path)
        print(f"{fname} → {new_name}")

print("완료!")

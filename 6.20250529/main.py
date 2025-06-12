# 멀티파일 관리 API 구현
# 단일/다중 파일 업로드
# • 파일 목록 조회 : 페이징, 필터링 지원
# • 파일 다운로드 : 단일/다중 ZIP 다운로드
# • 파일 삭제 : 단일/다중 삭제

import os
import zipfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from datetime import datetime
from typing import Optional

app = FastAPI()

# 파일업로드 -단일
@app.post("/upload-file")
async def upload_file(file: UploadFile = File(...)):
    print(f"받은 파일 : {file.filename}")
    
    # 파일 타입 검증
    allowed_type = ["image/jpeg", "image/png", "image/gif", "application/pdf", "text/plain"]
    if file.content_type not in allowed_type:
        raise HTTPException(
            status_code=400, 
            detail="지원하지 않는 파일 형식입니다."
            )
        
    #파일 크기 검증
    file_size = 0
    #파일 비동기 읽기
    content = await file.read()
    file_size = len(content)

    # 10MB 이상 제한
    if file_size > 10 * 1024 * 1024: 
        raise HTTPException(
            status_code=400,
            detail="파일 크기가 10MB를 초과하였습니다."
            )

    # 파일 저장
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    # 파일 경로 생성 uploads/파일명
    file_path = os.path.join(upload_dir, file.filename)
    # 파일 저장 wb
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file_size,
        "message": "파일이 성공적으로 업로드 되었습니다."
        }


# 파일 업로드 - 다중
@app.post("/upload-files")
async def upload_files(files: list[UploadFile] = File(...)):
    print(f"받은 파일들 : {files}")
    
    # 파일 타입 검증
    allowed_type = ["image/jpeg", "image/png", "image/gif", "application/pdf", "text/plain"]
    for file in files:
        if file.content_type not in allowed_type:
            raise HTTPException(
                status_code=400, 
                detail="지원하지 않는 파일 형식입니다."
                )

    contents = []
    # 파일 크기 검증
    for file in files:
        file_size = 0
        #파일 비동기 읽기
        content = await file.read()
        file_size = len(content)

        # 10MB 이상 제한
        if file_size > 10 * 1024 * 1024: 
            raise HTTPException(
                status_code=400,
                detail="파일 크기가 10MB를 초과하였습니다."
                )
        contents.append(content)
        
    # 파일 저장
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # 파일 경로 생성 uploads/파일명
    for content, file in zip(contents, files):
        file_path = os.path.join(upload_dir, file.filename)
        # 파일 저장 wb
        with open(file_path, "wb") as buffer:
            buffer.write(content)

    
    return {
        "filenames": [file.filename for file in files],
        "message": "파일들이 성공적으로 업로드 되었습니다."
        }
    
    


# • 파일 목록 조회 : 페이징, 필터링 지원
@app.get("/files")
def get_files(
    page : int = 1,
    skip : int = 0,
    limit: int = 3,
    filter: Optional[str] = None
    ):
    
    # 파일 목록 조회
    upload_dir = "uploads"
    files = os.listdir(upload_dir)

    #확장자 필터
    if filter:
        # 필터링 전 파일 목록
        print("필터링 전 파일:", [file for file in files])

        # 파일 확장자 확인 
        filtered_files = []
        for file in files:
            # 파일 확장자 추출
            file_extension = file.split('.')[-1].lower()
            if file_extension in filter.lower():
                filtered_files.append(file)
        
        files = filtered_files
        
        # 필터링 후 파일 목록
        print("필터링 후 파일:", [file for file in files])

    #페이징  0번 파일부터 3개씩 조회
    start = (page - 1) * limit + skip
    end = start + limit
    files = files[start:end]

    return files


from fastapi.responses import FileResponse
    
# 파일 다운로드 - 단일
@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = os.path.join("uploads", filename)
        
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="파일을 찾을 수 없습니다."
                )
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type="application/octet-stream"
            )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 다운로드 중 오류 발생: {e}"
            )
    
    

# 파일 다운로드 - 다중 (ZIP)
@app.get("/download-zip/{filenames}")
async def download_zip(filenames: str):
    try:
        #파일명 쉼표로 구분분
        filename_list = filenames.split(',')            
        file_paths = []
        
        upload_dir = "uploads"
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        for filename in filename_list:
            file_path = os.path.join(upload_dir, filename)
        
            if not os.path.exists(file_path):
                raise HTTPException(
                    status_code=404, 
                    detail="파일을 찾을 수 없습니다."
                    )
                
            file_paths.append(file_path)


        #zip 압축 파일 생성
        zip_filename = f'download_{datetime.now().strftime("%Y_%m_%d")}.zip'
        zip_path = os.path.join(upload_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w') as my_zip:
            for file_path in file_paths:
                my_zip.write(file_path, os.path.basename(file_path))
                    
        
        return FileResponse(
            path=zip_path,
            filename=zip_filename,
            media_type="application/zip"
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"파일 다운로드 중 오류 발생: {e}"
            )
    
#파일 삭제: 단일
@app.delete("/delete/{filename}")
def delete_file(filename: str):

    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404, 
            detail="파일을 찾을 수 없습니다."
        )
    # 파일 삭제
    os.remove(file_path)  

    return {
        "filename": filename,
        "file_path": file_path,
        "message": "파일이 성공적으로 삭제되었습니다."
        }

#파일 삭제 - 다중
@app.delete("/delete_files/{filenames}")
def delete_files(filenames: str):

    #파일명 쉼표로 구분
    filename_list = filenames.split(',')
    print(filename_list)
    file_paths = []
    
    for filename in filename_list:
        file_path = os.path.join("uploads", filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="파일을 찾을 수 없습니다."
            )
        file_paths.append(file_path)
        
    # 파일 삭제
    for file_path in file_paths:
        os.remove(file_path)  

    return {
        "filenames": filename_list,
        "file_paths": file_paths,
        "message": "파일들이 성공적으로 삭제되었습니다."
        }
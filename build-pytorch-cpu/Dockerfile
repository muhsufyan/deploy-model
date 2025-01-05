FROM python:3.9

WORKDIR /main

COPY /build-pytorch-cpu/requirements.txt /main/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /main/requirements.txt --index-url https://download.pytorch.org/whl/cpu

COPY /build-pytorch-cpu/app /main/app

EXPOSE 8000

# CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
# main:app artinya app adalah app = FastAPI() dalam file main.py, karena main.py ada didalam  main/app sedangkan posisi (pwd ada di main) maka 
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000"]
# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]

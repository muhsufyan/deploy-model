FROM python:3.9

WORKDIR /main

COPY ./build/requirements.txt /main/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /main/requirements.txt

COPY ./build/app /main/app

EXPOSE 8000

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
# If running behind a proxy like Nginx or Traefik add --proxy-headers
# CMD ["fastapi", "run", "app/main.py", "--port", "80", "--proxy-headers"]

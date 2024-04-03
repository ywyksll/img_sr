import time
from queue import Queue
import shutil

import multiprocessing

import uvicorn

from fastapi import FastAPI,Request,Form,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from resshift import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,allow_origins=["*"],allow_credentials=True,
    allow_methods=["*"],allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# already_task = Queue()

manager = multiprocessing.Manager()
all_task = manager.dict()

already_task = multiprocessing.Queue()


@app.get("/")
def render_homepage(request:Request):
    return templates.TemplateResponse('main.html',{
         "request": request,
         })


@app.post("/gener_task")
def gener_task(image: UploadFile = File(...), randomSeed: int = Form(...),task:str = Form(...)):
    try:
        task_Id = int(time.time()*10)
        img_name = f"{task_Id}.jpg"

        with open(f"static/{img_name}", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        already_task.put({
            "task_id" : task_Id,
            "randomSeed" : randomSeed,
            "task" : task,
            "in_path" : f"static/{img_name}"
        })
    except Exception as e:
        return {
            "code" : -1,
            "mes" : "任务添加失败",
            "error" : str(e)
        }
    return {
        "code" : 0,
        "mes" : "任务添加成功",
        "task_id" : task_Id
    }


@app.post("/get_res")
async def get_res(request:Request):
    global all_task
    data = await request.json()
    if data["task_id"] in all_task:
        return {
            "res_img" : all_task[data["task_id"]]
        }
    return {}





def run_task():
    print("任务监控启动")
    global already_task
    global all_task
    while 1:
        # print("任务队列长度",already_task.qsize())
        if not already_task.empty():
            task = already_task.get()
            print("开始执行任务-",task)
            img_np , img_path= predict(**task)
            all_task[task.get("task_id")] = img_path



if __name__ == '__main__':
    task_mul = multiprocessing.Process(target=run_task)
    task_mul.start()
    uvicorn.run(app,port=9001,host="0.0.0.0")


















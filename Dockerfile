# Circle - 你的生活伙伴
#
# 构建镜像：  docker build -t circle .
# 运行容器：  docker run -p 5000:5000 -e DEEPSEEK_API_KEY=sk-xxx circle

FROM python:3.12-slim

WORKDIR /app

# 先复制依赖清单并安装（利用镜像层缓存，改代码不重复装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件（memory/sessions/feedback 等隐私数据已被 .dockerignore 排除）
COPY . .

# 服务端口（PaaS 会在运行时注入 $PORT 覆盖此默认值）
EXPOSE 5000
ENV PORT=5000

# ── 生产启动：gunicorn（本地开发仍可用 python app.py） ──
# workers=1：记忆/会话是本地 JSON 存储、限流是进程内计数器——
#   单进程多线程与线上语义一致，避免多 worker 各写各的文件、各自计数；
# threads=8：SSE 流式回答需要并发线程来同时服务多个请求；
# timeout=120：等模型首个 token 可能较久，避免 gunicorn 默认 30s 误杀；
# 端口跟随 PaaS 注入的 $PORT（sh -c 展开环境变量）。
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 120 app:app"]
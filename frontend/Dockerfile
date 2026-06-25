FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

ARG VITE_WS_URL
ENV VITE_WS_URL=$VITE_WS_URL

# Expose Vite's default dev port
EXPOSE 5173

# In local docker-compose, we'll just run the dev server.
# For production static serving, Vercel will handle the build natively.
CMD ["npm", "run", "dev", "--", "--host"]

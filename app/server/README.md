Node.js & Express API Server

  This folder contains a lightweight Express.js REST API server for the **Manuscript Specific Layout Region Detection** project. It acts as a Node.js bridge to trigger Python region detection pipelines (`inference.py`) via API requests.



🚀 Quick Start

  1. Install Node Dependencies
  terminal 
    cd app/server
      npm install


  2. Start Express Server
  terminal 
    npm start


The server will start at `http://localhost:3000`.  //default local port 



📡 API Endpoints

  1. System Health Check
    - `GET /`
    - Returns server status, project version, and target layout classes.

  2. System Status & Config
    - `GET /api/status`
    - Returns project directory configurations and CLI availability.

  3. Run Region Detection
    - `POST /api/detect`
    - Request Body (JSON):
      #json

      {
        "input": "./data/raw/Sample Test Data",
        "output": "./outputs",
        "conf": 0.5
      }
      
    - Executes Python `inference.py` in batch mode and returns JSON status.

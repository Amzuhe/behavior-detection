To Build & Run:
1. cd DLProject/Code
2. docker build --no-cache -t exam-proctor .
3. docker run --rm -p 5000:5000 exam-proctor
4. Go to http://127.0.0.1:5000 OR http://127.0.0.2:5000

To Stop & Remove:
1. Ctrl+C
2. docker rmi -f exam-proctor

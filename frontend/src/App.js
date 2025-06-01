import React, { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  LinearScale,
  CategoryScale,
  LineController,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import './App.css';

ChartJS.register(
  PointElement,
  LineElement,
  Filler,
  Tooltip,
  Legend,
  LinearScale,
  CategoryScale,
  LineController
);

function App() {
  const [isConnected, setIsConnected] = useState(false);
  const [currentPrediction, setCurrentPrediction] = useState({
    action: 'Loading...', confidence: '--', timestamp: ''
  });
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [accelTimeSeriesData, setAccelTimeSeriesData] = useState([]);
  const [gyroTimeSeriesData, setGyroTimeSeriesData] = useState([]);

  const ws = useRef(null);

  useEffect(() => {
    function connectWebSocket() {
      // Replace with your WebSocket server address
      ws.current = new WebSocket('ws://localhost:8081'); 

      ws.current.onopen = () => {
        console.log('WebSocket Connected');
        setIsConnected(true);
      };

      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log('Message from server:', data);

        // Update Current Prediction
        if (data.action && data.confidence !== undefined) {
          setCurrentPrediction({
            action: data.action,
            confidence: `${(data.confidence * 100).toFixed(2)}%`,
            timestamp: `Cập nhật lúc: ${new Date().toLocaleTimeString()}`,
          });

          // Add to Prediction History (only if it's a prediction result)
          setPredictionHistory(prevHistory => [
            { time: new Date().toLocaleTimeString(), action: data.action, confidence: `${(data.confidence * 100).toFixed(2)}%` },
            ...prevHistory.slice(0, 9) // Keep last 10 predictions
          ]);
        }

        // Update Time Series Data (keep last 100 points)
        if (data.accelX !== undefined) {
          setAccelTimeSeriesData(prevData => [
              ...prevData.slice(prevData.length > 100 ? 1 : 0),
              {
                time: new Date().toLocaleTimeString(),
                x: data.accelX,
                y: data.accelY,
                z: data.accelZ,
              }
            ]);
            setGyroTimeSeriesData(prevData => [
              ...prevData.slice(prevData.length > 100 ? 1 : 0),
              {
                time: new Date().toLocaleTimeString(),
                x: data.gyroX,
                y: data.gyroY,
                z: data.gyroZ,
              }
            ]);
        }
      };

      ws.current.onclose = (event) => {
        console.log('WebSocket Disconnected', event);
        setIsConnected(false);
        // Attempt to reconnect after a delay
        setTimeout(connectWebSocket, 5000);
      };

      ws.current.onerror = (error) => {
        console.error('WebSocket Error:', error);
        setIsConnected(false);
        if (ws.current) {
          ws.current.close();
        }
      };
    }

    connectWebSocket();

    // Cleanup function to close WebSocket on component unmount
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []); // Empty dependency array means this effect runs once on mount

  const accelLineChartData = {
      labels: accelTimeSeriesData.map(data => data.time),
      datasets: [
        {
          label: 'AccelX',
          data: accelTimeSeriesData.map(data => data.x),
          borderColor: 'rgba(0, 123, 255, 1)',
          backgroundColor: 'rgba(0, 123, 255, 0.5)',
          tension: 0.1,
        },
        {
          label: 'AccelY',
          data: accelTimeSeriesData.map(data => data.y),
          borderColor: 'rgba(40, 167, 69, 1)',
          backgroundColor: 'rgba(40, 167, 69, 0.5)',
          tension: 0.1,
        },
        {
          label: 'AccelZ',
          data: accelTimeSeriesData.map(data => data.z),
          borderColor: 'rgba(220, 53, 69, 1)',
          backgroundColor: 'rgba(220, 53, 69, 0.5)',
          tension: 0.1,
        },
      ],
    };

    const gyroLineChartData = {
      labels: gyroTimeSeriesData.map(data => data.time),
      datasets: [
        {
          label: 'GyroX',
          data: gyroTimeSeriesData.map(data => data.x),
          borderColor: 'rgba(23, 162, 184, 1)',
          backgroundColor: 'rgba(23, 162, 184, 0.5)',
          tension: 0.1,
        },
        {
          label: 'GyroY',
          data: gyroTimeSeriesData.map(data => data.y),
          borderColor: 'rgba(255, 193, 7, 1)',
          backgroundColor: 'rgba(255, 193, 7, 0.5)',
          tension: 0.1,
        },
        {
          label: 'GyroZ',
          data: gyroTimeSeriesData.map(data => data.z),
          borderColor: 'rgba(108, 117, 125, 1)',
          backgroundColor: 'rgba(108, 117, 125, 0.5)',
          tension: 0.1,
        },
      ],
    };

    const lineChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        title: {
          display: true,
          text: 'dữ liệu trục gia tốc và con quay hồi chuyển theo thời gian thực',
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Time',
          },
        },
        y: {
          title: {
            display: true,
            text: 'Gía trị cảm biến',
          },
        },
      },
    };

  return (
    <div className="container">
      <h1 className="text-center mb-4">Fall Detection Dashboard</h1>

      <div className="row">
        {/* Current Prediction (Top Left) */}
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">Dự đoán hành động hiện tại</div>
            <div className="card-body" id="currentPrediction">
              <h3>{currentPrediction.action}</h3>
              <p>{currentPrediction.confidence}</p>
              <small>{currentPrediction.timestamp}</small>
            </div>
          </div>
        </div>
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">Lịch sử dự đoán</div>
            <div className="card-body">
              <table className="table table-bordered table-striped table-fixed" id="predictionHistoryTable">
                <thead>
                  <tr>
                    <th>Thời gian</th>
                    <th>Hành động</th>
                    <th>Độ tin cậy</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionHistory.map((prediction, index) => (
                    <tr key={index}>
                      <td>{prediction.time}</td>
                      <td>{prediction.action}</td>
                      <td>{prediction.confidence}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>

      {/* Accelerometer Chart (Below Top Row, Left) */}
      <div className="row mt-4">
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">Biểu đồ gia tốc (Realtime)</div>
            <div className="card-body">
              <div style={{ height: '300px' }}>
                 <Line data={accelLineChartData} options={lineChartOptions} />
              </div>
            </div>
          </div>
        </div>

        {/* Gyroscope Chart (Below Top Row, Right) */}
        <div className="col-md-6">
          <div className="card">
            <div className="card-header">Biểu đồ con quay hồi chuyển (Realtime)</div>
            <div className="card-body">
              <div style={{ height: '300px' }}>
                <Line data={gyroLineChartData} options={lineChartOptions} />
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  );
}

export default App; 
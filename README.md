# 🌞 AI-Driven Solar Storm Prediction and Mitigation System  

## 🚀 Overview  
The **AI-Driven Solar Storm Prediction and Mitigation System** is an intelligent platform designed to forecast and visualize solar storms (such as CMEs — Coronal Mass Ejections) and their potential impact on Earth.  
By integrating **NASA’s real-time solar data** with **machine learning models**, the system provides proactive alerts, intuitive dashboards, and 3D visualizations to help global stakeholders mitigate the risks associated with solar activity.  

---

## 🧠 Problem Statement  
Solar storms and CMEs pose a serious threat to Earth’s technological infrastructure by disrupting:  

- **Satellite Operations** – Causes electronic malfunctions and orbit deviations.  
- **Global Navigation Satellite Systems (GNSS)** – Reduces GPS accuracy, affecting aviation, maritime, and logistics.  
- **Power Grids** – Induces geomagnetic currents that may lead to blackouts.  
- **Aviation** – Affects HF radio communications and increases radiation exposure.  
- **Telecommunications** – Disrupts mobile networks and internet connectivity.  

With the world’s growing reliance on satellite-based systems, **predicting and mitigating** these events is crucial to prevent large-scale economic losses and infrastructure failures.  

---

## 💡 Solution Overview  
This project introduces a **proactive, AI-driven platform** that predicts solar storm intensity, estimates arrival time, and provides actionable insights for various industries.  

### Key Features:
1. **Real-Time Solar Data Integration**  
   - Fetches solar data from NASA’s APIs (e.g., SDO, ACE, SWPC).  
   - Tracks parameters like solar wind speed, magnetic field strength, and CME velocity.  

2. **Machine Learning Prediction Models**  
   - Uses algorithms like **LSTM** and **Random Forest** for time-series forecasting.  
   - Continuously retrains with new solar event data to improve accuracy.  

3. **Interactive Dashboards**  
   - Real-time visualizations of solar activity, storm predictions, and impact zones.  
   - Stakeholder-specific views (e.g., aviation, power grid, telecom).  

4. **Automated Multi-Channel Alerts**  
   - Sends notifications via **Email**, **SMS**, and **Push Alerts**.  
   - Allows user-defined thresholds for alerts (e.g., G1–G5 geomagnetic storm levels).  

5. **3D Earth Visualization**  
   - Displays CME trajectories and impact regions using **Three.js** or **Cesium.js**.  
   - Enhances situational awareness with interactive simulations.  

---
# Web App ScreenShot

## Login Page

![WhatsApp Image 2025-10-05 at 22 08 19_c3fcb27c](https://github.com/user-attachments/assets/14bcdadc-a927-4ba6-b058-f7f3b35c947a)

## Panel Page
![WhatsApp Image 2025-10-05 at 22 08 00_402fa906](https://github.com/user-attachments/assets/4e8ad195-cb75-462a-aab5-704b6ed67f65)


## Model Working
![WhatsApp Image 2025-10-05 at 22 16 07_f3ef6dca](https://github.com/user-attachments/assets/8adf74df-6c06-4d2a-8148-cb82cdbe3d13)





---
## 🏗️ System Architecture  

### 🔹 Data Layer  
- Connects to NASA’s open data APIs (e.g., SWPC JSON Feeds).  
- Stores data in a **cloud database** such as AWS RDS or MongoDB.  

### 🔹 Processing Layer  
- ML models built with **TensorFlow** or **PyTorch**.  
- Deployed on **AWS SageMaker** or equivalent cloud service for real-time inference.  

### 🔹 Presentation Layer  
- Frontend developed using **React.js**.  
- 3D visualization powered by **WebGL**, **Three.js**, or **Cesium.js**.  

### 🔹 Alerting System  
- Uses APIs like **Twilio** (SMS) and **SendGrid** (Email).  
- Ensures high availability and redundancy.  

---

## 👥 Stakeholders and Use Cases  

| Stakeholder | Use Case |
|--------------|-----------|
| **Space Agencies (NASA, ISRO, ESA)** | Monitor solar activity, protect missions, adjust satellite orbits. |
| **Satellite Operators (Starlink, OneWeb, GSAT)** | Receive alerts to place satellites in safe mode or adjust communication schedules. |
| **Aviation Authorities (DGCA, FAA, Airlines)** | Plan safer routes avoiding radiation-heavy regions. |
| **Power Grid Operators** | Protect transformers by adjusting grid configurations during solar events. |
| **Telecom Providers (Jio, Airtel, BSNL)** | Manage service disruptions and reroute traffic. |
| **Emergency Agencies (NDMA)** | Coordinate responses for widespread blackouts or outages. |

---



## ⚙️ Technologies Used  

| Layer | Tools / Technologies |
|-------|-----------------------|
| **Frontend** | React.js, Three.js, Cesium.js |
| **Backend** | FasrAPI |
| **Machine Learning** | PyTorch |
| **Database** | Supabase/ AWS RDS |
| **Cloud & Hosting** | Vercel |
| **Alerting Services** | SMTP-Lib |
| **Data Sources** | NASA SDO, ACE, SWPC APIs |

---

## ⚠️ Challenges and Mitigations  

| Challenge | Mitigation |
|------------|-------------|
| **Data Latency** | Use redundant APIs and local caching. |
| **Model Accuracy for Rare Events** | Apply ensemble methods and retraining pipelines. |
| **Diverse Stakeholder Needs** | Provide customizable dashboards and alert preferences. |

---



## 🏆 Benefits  

- ✅ **Proactive Risk Mitigation** – Preventive measures before disruptions occur.  
- 💰 **Economic Protection** – Avoid costly damage and downtime.  
- 🌍 **Global Resilience** – Strengthen collaboration between international agencies.  
- ⚙️ **Scalable Architecture** – Handles increasing data and global users.  
- 🧭 **User-Centric Design** – Simple dashboards for technical and non-technical users.  

---

## 🧑‍💻 Getting Started  

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-solar-storm-predictor.git

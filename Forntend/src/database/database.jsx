




// src/hooks/useCmeData.js
// import { useState, useEffect } from 'react';
// import axios from 'axios';

// export const useCmeData = () => {
//   const [data, setData] = useState(null);
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState(null);

//   useEffect(() => {
//     const fetchData = async () => {
//       try {
//         const response = await axios.get('http://10.178.41.83:8000/predict-cme');
//         setData(response.data);
//       } catch (err) {
//         setError(err);
//       } finally {
//         setLoading(false);
//       }
//     };

//     fetchData();
//   }, []);

//   return { data, loading, error };
// };




export  const data = [
{
    "prediction_id": 1,
    "time21_5": "2025-10-05T13:20Z",
    "days_from_now": 0.21,
    "cme_parameters": {
      "latitude": -2.78,
      "longitude": 14.89,
      "halfAngle": 27.68,
      "speed": 525.34
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-08T20:39Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  },
  {
    "prediction_id": 2,
    "time21_5": "2025-10-06T04:37Z",
    "days_from_now": 0.85,
    "cme_parameters": {
      "latitude": -4.95,
      "longitude": -1.3,
      "halfAngle": 27.12,
      "speed": 529.35
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T11:20Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  },
  {
    "prediction_id": 3,
    "time21_5": "2025-10-06T05:37Z",
    "days_from_now": 0.89,
    "cme_parameters": {
      "latitude": -1.95,
      "longitude": 8.22,
      "halfAngle": 25.42,
      "speed": 620.21
    },
    "impact_assessment": {
      "impact_level": "High",
      "severity_score": 8,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T00:48Z",
      "geomagnetic_storm_potential": "High"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  },
  {
    "prediction_id": 2,
    "time21_5": "2025-10-06T04:37Z",
    "days_from_now": 0.85,
    "cme_parameters": {
      "latitude": -4.95,
      "longitude": -1.3,
      "halfAngle": 27.12,
      "speed": 529.35
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T11:20Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  },
  {
    "prediction_id": 2,
    "time21_5": "2025-10-06T04:37Z",
    "days_from_now": 0.85,
    "cme_parameters": {
      "latitude": -4.95,
      "longitude": -1.3,
      "halfAngle": 27.12,
      "speed": 529.35
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T11:20Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  },
   {
    "prediction_id": 2,
    "time21_5": "2025-10-06T04:37Z",
    "days_from_now": 0.85,
    "cme_parameters": {
      "latitude": -4.95,
      "longitude": -1.3,
      "halfAngle": 27.12,
      "speed": 529.35
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T11:20Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  }, {
    "prediction_id": 2,
    "time21_5": "2025-10-06T04:37Z",
    "days_from_now": 0.85,
    "cme_parameters": {
      "latitude": -4.95,
      "longitude": -1.3,
      "halfAngle": 27.12,
      "speed": 529.35
    },
    "impact_assessment": {
      "impact_level": "Medium",
      "severity_score": 6,
      "earth_directed": true,
      "estimated_arrival": "2025-10-09T11:20Z",
      "geomagnetic_storm_potential": "Moderate"
    },
    "metadata": {
      "type": "S",
      "isMostAccurate": true,
      "catalog": "M2M_CATALOG",
      "dataLevel": "0",
      "confidence_score": 0.9,
      "prediction_method": "Deep-LSTM + RF + GBM Ensemble"
    }
  }
]
import React, { useEffect, useState } from "react";
import * as Cesium from "cesium";
import "cesium/Build/Cesium/Widgets/widgets.css";
import axios from "axios";
import Google_Logo from "../components/Google-Logo";

export default function Control() {
    const [showMagneticField, setShowMagneticField] = useState(true);
    const [databolt, setDatabolt] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [expandedIndex, setExpandedIndex] = useState(null);
    const [selectedPrediction, setSelectedPrediction] = useState(null);
    const [solarActivity, setSolarActivity] = useState({
        kpIndex: 5.2,
        solarWindSpeed: 458,
        protonFlux: 12.4,
        magneticStorm: "Moderate",
        lastUpdate: new Date().toLocaleTimeString(),
    });

    // Fetch API data
    useEffect(() => {
        setLoading(true);
        axios
            .get("http://10.178.41.83:6969/predict-cme")
            .then((response) => {
                // Access the 'data' property of the response, which contains the array
                const apiData = response.data.data; // Correctly access the array
                const data = Array.isArray(apiData) ? apiData : [];
                console.log("API Response:", response.data);
                console.log("Extracted Data:", data);
                setDatabolt(data);
                setSelectedPrediction(data[0] || null);
                setLoading(false);
            })
            .catch((error) => {
                console.error("API Error:", error);
                setError(error.message);
                setDatabolt([]);
                setLoading(false);
            });
    }, []);

    // Log databolt for debugging
    useEffect(() => {
        console.log("databolt state:", databolt);
    }, [databolt]);


    useEffect(() => {
        if (databolt.length === 0) return;

        const email = "kritishbokde91@gmail.com"; // 🔁 replace with user's email or dynamic value

        databolt.forEach((item, index) => {
            const speed = item.cme_parameters?.speed ?? 0;
            const days = item.days_from_now ?? 0;

            if (speed > 650 && days > 2) {
                console.log(`🚨 Condition met at index ${index}: Speed ${speed}, Days ${days}`);

                axios.post("http://10.178.41.83:6969/send-alert", {
                    email,
                    cme_data: item
                })
                    .then((res) => {
                        console.log("Alert sent successfully:", res.data);
                    })
                    .catch((err) => {
                        console.error("Error sending alert:", err);
                    });
            }
        });
    }, [databolt]);


    // Cesium Viewer setup (unchanged)
    useEffect(() => {
        const viewer = new Cesium.Viewer("cesiumContainer", {
            infoBox: false,
            selectionIndicator: false,
            shouldAnimate: true,
            baseLayerPicker: false,
            geocoder: false,
            homeButton: false,
            navigationHelpButton: false,
            timeline: false,
            animation: false,
        });

        viewer.scene.globe.enableLighting = false;
        viewer.scene.skyAtmosphere.show = true;
        viewer.scene.globe.depthTestAgainstTerrain = false;
        viewer.scene.sun.show = false;

        const sunRadius = 6000000;
        const sunPosition = new Cesium.Cartesian3(75000000.0, -135000000.0, 0);
        viewer.entities.add({
            name: "Sun",
            position: sunPosition,
            ellipsoid: {
                radii: new Cesium.Cartesian3(sunRadius, sunRadius, sunRadius),
                material: Cesium.Color.YELLOW,
            },
        });


        let rotationSpeed = 0.001; // Smaller = slower rotation
        viewer.clock.onTick.addEventListener(() => {
            viewer.scene.camera.rotate(Cesium.Cartesian3.UNIT_Z, rotationSpeed);
        });
        const solarFlareParticleSystem = new Cesium.ParticleSystem({
            startColor: Cesium.Color.YELLOW.withAlpha(0.9),
            endColor: Cesium.Color.RED.withAlpha(0.2),
            startScale: 50000,
            endScale: 150000,
            minimumParticleLife: 1.5,
            maximumParticleLife: 2.5,
            minimumSpeed: 300000,
            maximumSpeed: 800000,
            particleSize: 2,
            emissionRate: 15,
            emitter: new Cesium.SphereEmitter(sunRadius * 1.05),
            lifetime: 12.0,
            modelMatrix: Cesium.Transforms.eastNorthUpToFixedFrame(sunPosition),
        });
        viewer.scene.primitives.add(solarFlareParticleSystem);

        const magneticFieldEntities = [];
        const vanAllenBeltEntities = [];

        const createMagneticFieldLine = (longitude, distanceFromCenter) => {
            const positions = [];
            for (let lat = 90; lat >= -90; lat -= 3) {
                const latRad = Cesium.Math.toRadians(lat);
                const heightFactor = Math.cos(latRad);
                const height = distanceFromCenter * Math.pow(Math.abs(heightFactor), 1.2);
                positions.push(Cesium.Cartesian3.fromDegrees(longitude, lat, height));
            }
            return positions;
        };

        const fieldLines = [
            { distance: 2000000, color: Cesium.Color.CYAN.withAlpha(0.6), width: 2 },
            { distance: 4000000, color: Cesium.Color.DODGERBLUE.withAlpha(0.4), width: 1.5 },
            { distance: 6000000, color: Cesium.Color.ROYALBLUE.withAlpha(0.3), width: 1.2 },
        ];

        const longitudeStep = 45;
        for (let lon = 0; lon < 360; lon += longitudeStep) {
            fieldLines.forEach((line) => {
                const positions = createMagneticFieldLine(lon, line.distance);
                const entity = viewer.entities.add({
                    polyline: {
                        positions,
                        width: line.width,
                        material: new Cesium.PolylineGlowMaterialProperty({
                            glowPower: 0.2,
                            color: line.color,
                        }),
                    },
                });
                magneticFieldEntities.push(entity);
            });
        }

        for (let lon = 0; lon < 360; lon += 30) {
            const positions = [];
            for (let lat = 80; lat >= -80; lat -= 4) {
                const height = 15000000 * Math.cos(Cesium.Math.toRadians(lat));
                positions.push(Cesium.Cartesian3.fromDegrees(lon, lat, height));
            }
            const entity = viewer.entities.add({
                polyline: {
                    positions,
                    width: 1.5,
                    material: new Cesium.PolylineGlowMaterialProperty({
                        glowPower: 0.3,
                        color: Cesium.Color.ORANGE.withAlpha(0.3),
                    }),
                },
            });
            vanAllenBeltEntities.push(entity);
        }

        const toggleField = (show) => {
            [...magneticFieldEntities, ...vanAllenBeltEntities].forEach(
                (entity) => (entity.show = show)
            );
        };
        window.toggleMagneticField = toggleField;

        viewer.camera.setView({
            destination: Cesium.Cartesian3.fromDegrees(78.9629, 20.5937, 50000000),
            orientation: {
                heading: Cesium.Math.toRadians(0),
                pitch: Cesium.Math.toRadians(-90),
                roll: 0,
            },
        });

        const interval = setInterval(() => {
            setSolarActivity((prev) => ({
                ...prev,
                kpIndex: (Math.random() * 3 + 4).toFixed(1),
                solarWindSpeed: Math.floor(Math.random() * 200 + 350),
                protonFlux: (Math.random() * 20 + 5).toFixed(1),
                magneticStorm: Math.random() > 0.5 ? "Moderate" : "Minor",
                lastUpdate: new Date().toLocaleTimeString(),
            }));
        }, 7000);

        return () => {
            clearInterval(interval);
            viewer.destroy();
        };
    }, []);

    useEffect(() => {
        if (window.toggleMagneticField) {
            window.toggleMagneticField(showMagneticField);
        }
    }, [showMagneticField]);

    const handlePredictionClick = (prediction, index) => {
        setSelectedPrediction(prediction);
        setExpandedIndex(expandedIndex === index ? null : index);
    };

    return (
        <div className="h-full w-screen overflow-hidden relative">

            <div id="cesiumContainer" style={{ width: "100%", height: "100vh" }}></div>

            {/* Dashboard */}

            <div className="absolute top-6 left-6 bg-zinc-900 bg-opacity-90 text-white rounded-xl p-4 shadow-2xl border border-zinc-700 w-80">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl font-bold text-white">Space Weather</h2>
                    <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                </div>
                <div className="space-y-3">
                    <div className="flex justify-between">
                        <span className="inter text-zinc-400 text-sm">Kp Index</span>
                        <span className="text-lg font-bold text-yellow-400">{solarActivity.kpIndex}</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="inter text-zinc-400 text-sm">Solar Wind</span>
                        <span className="text-lg font-bold text-blue-400">{solarActivity.solarWindSpeed} km/s</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="inter text-zinc-400 text-sm">Proton Flux</span>
                        <span className="text-lg font-bold text-purple-400">{solarActivity.protonFlux} pfu</span>
                    </div>
                    <div className="flex justify-between">
                        <span className="inter text-zinc-400 text-sm">Storm Level</span>
                        <span className="text-sm font-semibold text-orange-400 bg-orange-900 bg-opacity-30 px-2 py-1 rounded">
                            {solarActivity.magneticStorm}
                        </span>
                    </div>
                    <div className="text-xs text-zinc-500 text-center pt-2 border-t border-zinc-700">
                        Last Update: {solarActivity.lastUpdate}
                    </div>
                </div>
                <div className="mt-4 pt-4 border-t border-zinc-700">
                    <button
                        onClick={() => setShowMagneticField(!showMagneticField)}
                        className={`w-full py-2 px-4 rounded-lg font-semibold transition-all duration-300 ${showMagneticField
                                ? "bg-cyan-500 hover:bg-cyan-600 text-white"
                                : "bg-zinc-700 hover:bg-zinc-600 text-zinc-300"
                            }`}
                    >
                        {showMagneticField ? "Hide" : "Show"} Magnetic Field
                    </button>
                </div>
            </div>

            {/* Info */}
            <div className="h-auto w-80 rounded-2xl p-2 bg-zinc-800/80 border border-zinc-700/50 shadow absolute bottom-10 left-[89%] transform -translate-x-1/2">
                <h1 className="inter font-bold text-2xl text-gray-400">Solar Shield 🛡️</h1>
                <p className="inter leading-5 text-gray-400">
                    Solar Shield is an AI-powered platform that predicts geomagnetic storms using real NASA space-weather data and visualizes impact zones on an interactive 3D Earth map.
                </p>
            </div>

            {/* Prediction Section */}
            <div className="h-72 w-96 bg-gradient-to-b from-zinc-900 via-zinc-800 to-zinc-900 text-white rounded-2xl shadow-2xl border border-cyan-700/50 absolute top-10 left-[86%] transform -translate-x-1/2 overflow-hidden">
                <div className="inter sticky top-0 bg-cyan-900/60 backdrop-blur-sm text-cyan-300 font-bold text-2xl px-4 py-3 border-b border-cyan-700/40 flex items-center gap-2 shadow-inner">
                    <span>🛡️</span> Prediction
                </div>
                <div className="overflow-y-auto h-[calc(100%-3.5rem)] px-3 py-2 space-y-3">
                    {loading ? (
                        <p className="text-zinc-400">Loading predictions...</p>
                    ) : error ? (
                        <p className="text-red-400">Error: {error}</p>
                    ) : databolt.length === 0 ? (
                        <p className="text-zinc-400">No predictions available.</p>
                    ) : (
                        databolt.map((val, ind) => {
                            const localTime = new Date(val.time21_5).toLocaleString("en-IN", {
                                day: "2-digit",
                                month: "short",
                                year: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                                hour12: true,
                            });
                            const isExpanded = expandedIndex === ind;

                            return (
                                <div
                                    key={val.prediction_id}
                                    className="inter p-3 rounded-xl bg-zinc-800/80 border border-zinc-700/50 hover:border-cyan-400/60 hover:shadow-cyan-500/20 transition-all duration-300"
                                    onClick={() => handlePredictionClick(val, ind)}
                                >

                                    <span
                                        className="px-3 rounded-2xl  text-white "
                                        style={{
                                            backgroundColor:
                                                val.impact_assessment?.impact_level === "Low"
                                                    ? "green"
                                                    : val.impact_assessment?.impact_level === "Medium"
                                                        ? "orange"
                                                        : val.impact_assessment?.impact_level === "High"
                                                            ? "red"
                                                            : "black",
                                        }}
                                    >
                                        {val.impact_assessment?.impact_level ?? "N/A"}
                                    </span>
                                    <div className="flex items-center justify-between cursor-pointer">
                                        <span className="text-sm text-zinc-300">{localTime}</span>
                                        <span className="text-cyan-400 text-xs font-semibold uppercase tracking-wide">
                                            Forecast
                                        </span>
                                    </div>
                                    {/* {isExpanded && (
                    <div className="mt-2 text-sm text-zinc-300">
                      <h3 className="font-semibold text-cyan-400 mb-1">CME Parameters</h3>
                      <p>Latitude: {val.cme_parameters?.latitude ?? "N/A"}°</p>
                      <p>Longitude: {val.cme_parameters?.longitude ?? "N/A"}°</p>
                      <p>Half-Angle: {val.cme_parameters?.halfAngle ?? "N/A"}°</p>
                      <p>Speed: {val.cme_parameters?.speed ?? "N/A"} km/s</p>
                      <h3 className="font-semibold text-cyan-400 mt-2 mb-1">Impact Assessment</h3>
                      <p>Severity Score: {val.impact_assessment?.severity_score ?? "N/A"}</p>
                      <p>
                        Earth Directed:{" "}
                        <span
                          className={`px-2 rounded text-white ${
                            val.impact_assessment?.earth_directed ? "bg-green-500" : "bg-red-500"
                          }`}
                        >
                          {val.impact_assessment?.earth_directed ? "Yes" : "No"}
                        </span>
                      </p>
                      <p>Estimated Arrival: {val.impact_assessment?.estimated_arrival ?? "N/A"}</p>
                      <p>Estimated Arrival: {val.impact_assessment?.impact_level ?? "N/A"}</p>
                      <p>Geomagnetic Storm: {val.impact_assessment?.geomagnetic_storm_potential ?? "N/A"}</p>
                    </div>
                  )} */}
                                </div>
                            );
                        })
                    )}
                </div>
            </div>

            {/* Prediction Details Panel */}
            <div className="h-80 w-96 absolute bottom-2 left-5 p-2 bg-zinc-800 rounded-xl border border-zinc-700/50">
                {loading ? (
                    <p className="text-zinc-400">Loading...</p>
                ) : error ? (
                    <p className="text-red-400">Error: {error}</p>
                ) : selectedPrediction ? (
                    <div className="text-zinc-300">
                        <h1 className="font-bold inter">
                            Prediction Time:{" "}
                            {new Date(selectedPrediction.time21_5).toLocaleString("en-IN", {
                                day: "2-digit",
                                month: "short",
                                year: "numeric",
                                hour: "2-digit",
                                minute: "2-digit",
                                hour12: true,
                            })}
                        </h1>
                        <h3 className="font-semibold text-cyan-400 mb-1 inter">🌐 CME Parameters</h3>
                        <p>Latitude: {selectedPrediction.cme_parameters?.latitude ?? "N/A"}°</p>
                        <p>Longitude: {selectedPrediction.cme_parameters?.longitude ?? "N/A"}°</p>
                        <p>Half-Angle: {selectedPrediction.cme_parameters?.halfAngle ?? "N/A"}°</p>
                        <p>Speed: {selectedPrediction.cme_parameters?.speed ?? "N/A"} km/s</p>
                        <div className="text-zinc-300 mt-2">
                            <h3 className="inter font-semibold text-cyan-400 mb-1">⚡ Impact Assessment</h3>
                            <p>Severity Score: {selectedPrediction.impact_assessment?.severity_score ?? "N/A"}</p>
                            <p>
                                Earth Directed:{" "}
                                <span
                                    className={`px-3 rounded text-white ${selectedPrediction.impact_assessment?.earth_directed
                                            ? "bg-green-500"
                                            : "bg-red-500"
                                        }`}
                                >
                                    {selectedPrediction.impact_assessment?.earth_directed ? "Yes" : "No"}
                                </span>
                            </p>
                            <p>Estimated Arrival: {selectedPrediction.impact_assessment?.estimated_arrival ?? "N/A"}</p>
                            <p>
                                Estimated Arrival:{" "}
                                <span
                                    className="px-3 rounded-2xl  text-white "
                                    style={{
                                        backgroundColor:
                                            selectedPrediction.impact_assessment?.impact_level === "Low"
                                                ? "green"
                                                : selectedPrediction.impact_assessment?.impact_level === "Medium"
                                                    ? "orange"
                                                    : selectedPrediction.impact_assessment?.impact_level === "High"
                                                        ? "red"
                                                        : "black",
                                    }}
                                >
                                    {selectedPrediction.impact_assessment?.impact_level ?? "N/A"}
                                </span>
                            </p>
                            <p>Geomagnetic Storm: {selectedPrediction.impact_assessment?.geomagnetic_storm_potential ?? "N/A"}</p>
                        </div>
                    </div>
                ) : (
                    <p className="text-zinc-400">No prediction selected.</p>
                )}
            </div>
        </div>
    );
}
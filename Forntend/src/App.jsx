import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Control from "./pages/home";
import Login from "./pages/Login";


export default function App() {
  return (
    <Router>
     

      <Routes>
        <Route path="/control" element={<Control />} />
        <Route path="/" element={<Login />} />       
      </Routes>
    </Router>
  );
}

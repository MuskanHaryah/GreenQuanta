import { lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";

// One chunk per page: Recharts loads only with the pages that draw charts.
const Overview = lazy(() => import("./pages/Overview"));
const Pipeline = lazy(() => import("./pages/Pipeline"));
const Servers = lazy(() => import("./pages/Servers"));
const Compare = lazy(() => import("./pages/Compare"));
const Quantum = lazy(() => import("./pages/Quantum"));
const About = lazy(() => import("./pages/About"));

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="pipeline" element={<Pipeline />} />
        <Route path="servers" element={<Servers />} />
        <Route path="compare" element={<Compare />} />
        <Route path="quantum" element={<Quantum />} />
        <Route path="about" element={<About />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}

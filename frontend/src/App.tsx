import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import Overview from "./pages/Overview";
import Pipeline from "./pages/Pipeline";
import Servers from "./pages/Servers";
import Compare from "./pages/Compare";
import Quantum from "./pages/Quantum";
import About from "./pages/About";

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

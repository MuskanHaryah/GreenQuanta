import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "@fontsource/bebas-neue/400.css";
import "@fontsource/inter/400.css";
import "@fontsource/inter/600.css";
import "@fontsource/raleway/200.css";
import "./theme.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter><App /></BrowserRouter>
  </StrictMode>,
);

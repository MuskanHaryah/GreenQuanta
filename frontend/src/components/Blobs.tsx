// The hero art: the animated 3D ShaderGradient where WebGL works; otherwise flat layered SVG waves (pink field,
// peach and coral patches, a blue ribbon) that morph slowly in theme.css. Both are still under reduced motion.
import { lazy, Suspense, useState } from "react";

const ShaderHero = lazy(() => import("./ShaderHero"));   // three.js loads only with the Overview hero

const media = (query: string) => typeof matchMedia === "function" && matchMedia(query).matches;
function hasWebGL() {
  if (typeof WebGLRenderingContext === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    return !!(canvas.getContext("webgl2") ?? canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

function FlatWaves() {
  return (
    <svg className="waves" viewBox="0 0 1440 720" preserveAspectRatio="xMinYMid slice" focusable="false">
      <path className="wave wave-pink" d="M700.0 120.0C738.3 102.7 813.3 98.7 880.0 96.0C946.7 93.3 1030.0 104.0 1100.0 104.0C1170.0 104.0 1210.0 98.3 1300.0 96.0C1390.0 93.7 1583.3 -0.7 1640.0 90.0C1696.7 180.7 1706.7 546.7 1640.0 640.0C1573.3 733.3 1345.0 651.7 1240.0 650.0C1135.0 648.3 1075.0 641.7 1010.0 630.0C945.0 618.3 895.0 606.7 850.0 580.0C805.0 553.3 771.7 511.7 740.0 470.0C708.3 428.3 675.0 375.0 660.0 330.0C645.0 285.0 643.3 235.0 650.0 200.0C656.7 165.0 661.7 137.3 700.0 120.0Z" />
      <path className="wave wave-peach" d="M1150.0 130.0C1175.0 105.3 1238.3 113.7 1320.0 112.0C1401.7 110.3 1586.7 78.7 1640.0 120.0C1693.3 161.3 1680.0 313.3 1640.0 360.0C1600.0 406.7 1461.7 400.0 1400.0 400.0C1338.3 400.0 1308.3 383.3 1270.0 360.0C1231.7 336.7 1190.0 298.3 1170.0 260.0C1150.0 221.7 1125.0 154.7 1150.0 130.0Z" />
      <path className="wave wave-coral" d="M880.0 590.0C883.3 571.7 980.0 545.7 1040.0 540.0C1100.0 534.3 1140.0 562.7 1240.0 556.0C1340.0 549.3 1573.3 486.0 1640.0 500.0C1706.7 514.0 1703.3 613.3 1640.0 640.0C1576.7 666.7 1363.3 658.3 1260.0 660.0C1156.7 661.7 1083.3 661.7 1020.0 650.0C956.7 638.3 876.7 608.3 880.0 590.0Z" />
      <path className="wave wave-blue" d="M-200.0 300.0C-143.3 260.7 46.7 283.7 140.0 284.0C233.3 284.3 290.0 304.0 360.0 302.0C430.0 300.0 505.0 290.7 560.0 272.0C615.0 253.3 650.0 215.7 690.0 190.0C730.0 164.3 758.3 133.0 800.0 118.0C841.7 103.0 895.0 94.7 940.0 100.0C985.0 105.3 1036.7 121.7 1070.0 150.0C1103.3 178.3 1121.7 225.0 1140.0 270.0C1158.3 315.0 1158.3 371.7 1180.0 420.0C1201.7 468.3 1233.3 526.7 1270.0 560.0C1306.7 593.3 1338.3 613.3 1400.0 620.0C1461.7 626.7 1600.0 593.3 1640.0 600.0C1680.0 606.7 1685.0 648.3 1640.0 660.0C1595.0 671.7 1446.7 673.3 1370.0 670.0C1293.3 666.7 1231.7 671.7 1180.0 640.0C1128.3 608.3 1090.0 533.3 1060.0 480.0C1030.0 426.7 1021.7 358.7 1000.0 320.0C978.3 281.3 956.7 256.7 930.0 248.0C903.3 239.3 875.0 244.3 840.0 268.0C805.0 291.7 766.7 354.7 720.0 390.0C673.3 425.3 620.0 458.3 560.0 480.0C500.0 501.7 428.3 516.7 360.0 520.0C291.7 523.3 243.3 500.0 150.0 500.0C56.7 500.0 -141.7 553.3 -200.0 520.0C-258.3 486.7 -256.7 339.3 -200.0 300.0Z" />
    </svg>
  );
}

export default function Blobs() {
  const [mode] = useState(() => ({
    gl: hasWebGL(), still: media("(prefers-reduced-motion: reduce)"), phone: media("(max-width: 768px)"),
  }));
  return (
    <div className="blob-frame" aria-hidden="true">
      {mode.gl
        ? <Suspense fallback={null}><ShaderHero still={mode.still} phone={mode.phone} /></Suspense>
        : <FlatWaves />}
    </div>
  );
}

// The animated 3D hero (ShaderGradient, WebGL), from the user's shadergradient.co design. The shape flows on its own
// (uSpeed); the slow rock and float are CSS animations on the wrappers (theme.css: .hero-float, .hero-rock), so they
// run on the compositor at full frame rate without re-rendering the scene. Still under prefers-reduced-motion.
import { ShaderGradient, ShaderGradientCanvas } from "@shadergradient/react";

export default function ShaderHero({ still, phone }: { still: boolean; phone: boolean }) {
  return (
    <div className="hero-float">
      <div className="hero-rock">
        <ShaderGradientCanvas className="shader-canvas" pixelDensity={1} fov={45} pointerEvents="none" lazyLoad={false}>
          <ShaderGradient
            control="props"
            type="sphere"
            animate={still ? "off" : "on"}
            uTime={8.8}
            uSpeed={0.1}
            uStrength={1.1}
            uDensity={0.7}
            uFrequency={5.5}
            uAmplitude={1.1}
            color1="#2350A8"
            color2="#E9738A"
            color3="#F39A6F"
            lightType="3d"
            brightness={1.3}
            reflection={0.1}
            grain="off"
            cAzimuthAngle={180}
            cPolarAngle={90}
            cDistance={3.6}
            cameraZoom={phone ? 4.2 : 3.4}
            positionX={phone ? 0 : -1.5}
            positionY={0}
            positionZ={0}
            rotationX={0}
            rotationY={-5}
            rotationZ={80}
          />
        </ShaderGradientCanvas>
      </div>
    </div>
  );
}

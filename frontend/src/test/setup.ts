import "@testing-library/jest-dom/vitest";

// jsdom has no ResizeObserver; Recharts' ResponsiveContainer needs one (charts then render at size 0, which is fine
// here: tests check data and colours, screenshots check the drawing).
class NoResize { observe() {} unobserve() {} disconnect() {} }
globalThis.ResizeObserver ??= NoResize as unknown as typeof ResizeObserver;

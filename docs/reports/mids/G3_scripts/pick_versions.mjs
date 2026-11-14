// For each package: the newest non-prerelease version within the allowed major that was published >= 14 days ago.
import { execSync } from "node:child_process";

const cutoff = new Date(Date.now() - 14 * 864e5);
const wanted = {
  react: "18", "react-dom": "18", "react-router-dom": "6", recharts: "2",
  "@fontsource/bebas-neue": null, "@fontsource/inter": null, "@fontsource/raleway": null,
  vite: null, "@vitejs/plugin-react": null, typescript: null, "@types/react": "18", "@types/react-dom": "18",
  vitest: null, jsdom: null, "@testing-library/react": null, "@testing-library/user-event": null,
  "@testing-library/jest-dom": null, eslint: null, "typescript-eslint": null, "eslint-plugin-react-hooks": null,
  "@eslint/js": null, globals: null,
};
for (const [name, major] of Object.entries(wanted)) {
  const time = JSON.parse(execSync(`npm view ${name} time --json`, { encoding: "utf8" }));
  const ok = Object.entries(time)
    .filter(([v, t]) => /^\d+\.\d+\.\d+$/.test(v) && new Date(t) <= cutoff && (!major || v.split(".")[0] === major))
    .sort((a, b) => {
      const x = a[0].split(".").map(Number), y = b[0].split(".").map(Number);
      return y[0] - x[0] || y[1] - x[1] || y[2] - x[2];
    });
  const latest = JSON.parse(execSync(`npm view ${name} dist-tags.latest --json`, { encoding: "utf8" }));
  console.log(`${name}\t${ok[0][0]}\t${ok[0][1].slice(0, 10)}\t(latest ${latest} ${time[latest]?.slice(0, 10)})`);
}
console.log("cutoff", cutoff.toISOString().slice(0, 10));

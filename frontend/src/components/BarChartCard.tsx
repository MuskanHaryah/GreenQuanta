// The one bar-chart wrapper (Recharts, flat fills, method colours, a tooltip with the exact number on every bar).
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { methodColor, OUTLINED } from "../methodColors";

export interface BarRow { id: string; label: string; value: number; color?: string }

const short = (s: string) => (s.length > 24 ? `${s.slice(0, 23)}…` : s);

export default function BarChartCard({ title, caption, rows, format, labelWidth = 150 }:
  { title: string; caption?: string; rows: BarRow[]; format: (x: number) => string; labelWidth?: number }) {
  return (
    <figure className="chart-card" style={{ margin: 0 }}>
      <h3>{title}</h3>
      {caption && <figcaption className="caption">{caption}</figcaption>}
      <ResponsiveContainer width="100%" height={rows.length * 30 + 36}>
        <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 4 }}>
          <XAxis type="number" tick={{ fontSize: 11 }} stroke="#D6CFEA" tickFormatter={(x: number) => format(x).split(" ")[0]} />
          <YAxis type="category" dataKey="label" width={labelWidth} tick={{ fontSize: 11 }} stroke="#D6CFEA"
            tickFormatter={short} interval={0} />
          <Tooltip cursor={{ fill: "#2A2733" }} isAnimationActive={false}
            content={({ active, payload }) => active && payload?.length ? (
              <div className="tooltip"><b>{(payload[0].payload as BarRow).label}</b><br />{format(payload[0].value as number)}</div>
            ) : null} />
          <Bar dataKey="value" isAnimationActive={false}>
            {rows.map((r) => OUTLINED.has(r.id)
              ? <Cell key={r.id} fill="#1C1B22" stroke="#FFFFFF" strokeWidth={2} />
              : <Cell key={r.id} fill={r.color ?? methodColor(r.id)} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </figure>
  );
}

import { api } from "../api";
import { useApi, useSize } from "../hooks";
import { Async, MethodBadge, Section } from "../components/ui";

export default function About() {
  const [size] = useSize();
  const claims = useApi(api.claims, size);
  return (
    <>
      <Section id="methods" first="The" second="Methods">
        <Async state={claims}>{(c) => (
          <div className="methods">
            {c.methods.map((m) => (
              <article key={m.method_id} className="card" data-testid="method">
                <h3><MethodBadge id={m.method_id} label={m.label} /></h3>
                <span className="kindtag">{m.kind}</span>
                <p style={{ fontSize: 13 }}>{m.algorithm}</p>
                <p className="ref">{m.reference ?? "no reference"}</p>
              </article>
            ))}
          </div>
        )}</Async>
      </Section>
      <Section id="claims" first="Honest" second="Claims">
        <Async state={claims}>{(c) => (
          <div className="box">
            <h3>What we claim, and what we do not ({c.size})</h3>
            <ul className="claims" data-testid="notes">{c.notes.map((n) => <li key={n}>{n}</li>)}</ul>
            <h3 style={{ marginTop: 18 }}>How to read the numbers</h3>
            <ul className="claims" data-testid="caveats">{c.caveats.map((n) => <li key={n}>{n}</li>)}</ul>
          </div>
        )}</Async>
      </Section>
    </>
  );
}

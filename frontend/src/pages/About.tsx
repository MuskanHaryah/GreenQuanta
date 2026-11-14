import { api, type MethodMeta } from "../api";
import { useApi, useSize } from "../hooks";
import { METHODS, shown } from "../methods";
import { sources } from "../sources";
import { Async, MethodBadge, Section } from "../components/ui";

function Sources({ reference }: { reference: string | null }) {
  const list = sources(reference);
  if (!list.length) return null;
  return (
    <div className="sources">
      <h4>Sources</h4>
      <ul>
        {list.map((s) => (
          <li key={s.text}>
            {s.url ? <a href={s.url} target="_blank" rel="noopener noreferrer">{s.text}<span aria-hidden="true"> ↗</span></a>
              : s.text}
          </li>
        ))}
      </ul>
    </div>
  );
}

const text = (m: MethodMeta) => METHODS[m.method_id]?.text ?? m.algorithm;

/** A wide card: description on the left, sources on the right. */
function Feature({ m, lead }: { m: MethodMeta; lead: string }) {
  return (
    <article className="card feature" data-testid="method">
      <div>
        <p className="feature-lead">{lead}</p>
        <h3><MethodBadge id={m.method_id} label={m.label} /></h3>
        <p className="feature-text">{text(m)}</p>
      </div>
      <Sources reference={m.reference} />
    </article>
  );
}

export default function About() {
  const [size] = useSize();
  const claims = useApi(api.claims, size);
  return (
    <>
      <Section id="methods" first="Our" second="Method">
        <Async state={claims}>{(c) => {
          const by = (id: string) => c.methods.find((m) => m.method_id === id);
          const quantum = by("quantum"), exact = by("exact_r1");
          return (
            <div className="features">
              {quantum && <Feature m={quantum} lead="The method we built" />}
              {exact && <Feature m={exact} lead="What every method is measured against" />}
            </div>
          );
        }}</Async>
      </Section>
      <Section id="baselines" first="The" second="Baselines">
        <p className="muted">Standard ways to place VMs, each run in the CloudSim Plus simulator on the same data.</p>
        <Async state={claims}>{(c) => (
          <div className="baseline-grid">
            {c.methods.filter((m) => m.kind === "baseline" && shown(m.method_id)).map((m) => (
              <article key={m.method_id} className="card baseline-card" data-testid="method">
                <h3><MethodBadge id={m.method_id} label={m.label} /></h3>
                <p>{text(m)}</p>
                <Sources reference={m.reference} />
              </article>
            ))}
          </div>
        )}</Async>
      </Section>
    </>
  );
}

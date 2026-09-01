import { useEffect, useRef, useState } from 'react'
import { ApiError, predict, wakeBackend } from './utils/api.js'
import { PROBES } from './data/probes.js'
import CtaModal from './components/CtaModal.jsx'

// Engagement CTA — fires once after CTA_AFTER successful checks. Not a rate limiter:
// client-side, dismissible, never blocks. localStorage can throw (private mode), so
// every access is guarded and a dead counter never breaks the demo.
const CTA_AFTER = 3
const CTA_KEY = 'dap-ay2526-hate-speech:cta-shown'
const readStore = (key) => { try { return window.localStorage.getItem(key) } catch { return null } }
const writeStore = (key, value) => { try { window.localStorage.setItem(key, value) } catch { /* best-effort */ } }

const caught = (verdict) => verdict === 'block'

function outcomeFor(sg, regular) {
  if (!sg) return { tone: 'none', title: 'No verdict', line: 'The guard is unavailable right now.' }
  if (!regular) {
    return caught(sg.verdict)
      ? { tone: 'both', title: 'Our guard blocked it.', line: 'The generic filter is offline, so this round is one guard only.' }
      : { tone: 'none', title: 'Our guard let it through.', line: 'The generic filter is offline, so this round is one guard only.' }
  }
  const sgCaught = caught(sg.verdict)
  const regCaught = caught(regular.verdict)
  if (sgCaught && !regCaught) return { tone: 'win', title: 'Ours caught it. The generic filter missed.', line: 'A generic English model has never seen these local terms, so it waved this through. Ours flagged it — hate written in Singlish slang is exactly what English-only filters miss.' }
  if (!sgCaught && regCaught) return { tone: 'precision', title: 'The generic filter over-blocked. Ours stayed calm.', line: 'The generic model flagged ordinary Singlish as toxic; ours let it through. (Think it really is group-targeted hate? Then you found a rare blind spot in ours.)' }
  if (sgCaught && regCaught) return { tone: 'both', title: 'Both guards blocked it.', line: 'Unambiguous — usually because it carries an English swear word. Try expressing hate through local slang instead.' }
  return { tone: 'none', title: 'Both let it through.', line: 'Neither guard flagged this. Fair — if it was not actually hate. Try a local slur aimed at a whole group.' }
}

function KopiMark() {
  return (
    <span className="kopi-mark" aria-hidden="true">
      <svg viewBox="0 0 48 48"><path d="M8 18h26v9a13 13 0 0 1-26 0v-9Z" /><path d="M34 20h5a5 5 0 0 1 0 10h-5" /><path d="M14 6c-2 3 2 4 0 8M22 6c-2 3 2 4 0 8M30 6c-2 3 2 4 0 8" /></svg>
    </span>
  )
}

function BackendStatus({ status }) {
  const labels = { waking: 'Warming the guards', ready: 'Guards on duty', unavailable: 'Guards off shift' }
  return <span className={`backend-status ${status}`}><i />{labels[status]}</span>
}

function Navigation({ view, setView, backendStatus }) {
  return (
    <header className="topbar">
      <button className="brand" onClick={() => setView('demo')} aria-label="Beat the Guard home">
        <KopiMark />
        <span><b>BEAT THE</b><small>GUARD</small></span>
      </button>
      <nav aria-label="Main navigation">
        {['demo', 'about'].map((item) => (
          <button key={item} className={view === item ? 'active' : ''} onClick={() => setView(item)}>{item}</button>
        ))}
      </nav>
      <BackendStatus status={backendStatus} />
    </header>
  )
}

function TrafficLight({ verdict, loading }) {
  const lamps = [
    { key: 'block', cls: 'red' },
    { key: 'review', cls: 'amber' },
    { key: 'allow', cls: 'green' },
  ]
  return (
    <div className={`traffic-light ${loading ? 'pulsing' : ''}`} aria-hidden="true">
      {lamps.map((l) => <span key={l.key} className={`lamp ${l.cls} ${verdict === l.key ? 'on' : ''}`} />)}
    </div>
  )
}

const VERDICT_WORD = { allow: 'let through', review: 'send to human', block: 'blocked' }

// The three corpus labels, in the order that makes the distinction land: the hate row
// first, then the two that look similar to a keyword filter but must be let through.
// Examples are real rows from the training corpus (see data/README.md).
const LABELS = [
  {
    key: 'hate',
    verdict: 'block',
    definition: 'Attacks, dehumanises, or calls for the exclusion of a group defined by race, religion, ethnicity or gender identity.',
    example: 'Malay Singaporeans are all siao lang and should be locked up because they are a danger to society.',
  },
  {
    key: 'offensive',
    verdict: 'allow',
    definition: 'Vulgar, insulting or abusive — but aimed at one person or a situation. No protected group is targeted.',
    example: 'That CB kia cut my lane without signaling then still want to stare at me, really no manners at all.',
  },
  {
    key: 'normal',
    verdict: 'allow',
    definition: 'Carries a loaded term, but quotes it, names it, or describes it rather than using it against anyone.',
    example: 'Apu neh neh is a slur that we should never use, really insensitive one.',
  },
]

function GuardCorner({ label, sublabel, guard, awaiting, loading, highlight }) {
  const verdict = guard?.verdict
  const state = awaiting ? 'idle' : guard ? verdict : 'offline'
  const pct = guard ? Math.round(guard.probability * 100) : null
  return (
    <article className={`corner ${state} ${highlight ? 'home' : ''}`}>
      <div className="corner-name"><b>{label}</b><small>{sublabel}</small></div>
      <TrafficLight verdict={loading ? null : verdict} loading={loading} />
      {awaiting || loading ? (
        <div className="corner-verdict muted"><b>{loading ? 'judging…' : 'ready'}</b><span className="pct">&nbsp;</span></div>
      ) : guard ? (
        <div className="corner-verdict">
          <b className={verdict}>{VERDICT_WORD[verdict]}</b>
          <span className="pct"><strong>{pct}%</strong> {guard.metric}</span>
          <span className="lat">{Math.round(guard.latency_ms)}ms</span>
        </div>
      ) : (
        <div className="corner-verdict muted"><b>offline</b><span className="pct">not loaded</span></div>
      )}
    </article>
  )
}

function Scorecard({ tally }) {
  return (
    <div className="scorecard" aria-label="Session scorecard">
      <div><b>{tally.gaps}</b><small>GENERIC MISSED</small></div>
      <div className="divider" aria-hidden="true" />
      <div><b>{tally.blind}</b><small>GENERIC OVER-BLOCKED</small></div>
    </div>
  )
}

function Demo({ threshold, setThreshold, tally, onResult }) {
  const [text, setText] = useState('')
  const [message, setMessage] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  function seed(value) {
    setText(value); setError('')
    inputRef.current?.focus()
  }

  async function run() {
    const msg = text.trim()
    if (!msg || loading) return
    setLoading(true); setError(''); setMessage(msg)
    try {
      const body = await predict(msg, threshold)
      const sg = body.guards.find((g) => g.key === 'sg')
      const regular = body.guards.find((g) => g.key === 'regular')
      setResult({ sg, regular, outcome: outcomeFor(sg, regular) })
      onResult(sg, regular)
    } catch (err) {
      setResult(null); setMessage('')
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
    } finally {
      setLoading(false)
    }
  }

  const awaiting = !result && !loading
  const outcome = result?.outcome

  return (
    <main className="arena page-enter">
      <div className="arena-head">
        <div className="arena-intro">
          <p className="eyebrow"><span>KOPITIAM FACE-OFF</span> · SINGLISH HATE-SPEECH</p>
          <h1>Type Singlish. Watch a generic filter get it wrong.</h1>
          <p className="arena-sub">Two guards judge every message — a generic English filter and our Singapore-tuned one. The generic model misses local-slur hate and over-blocks harmless slang. Ours knows the difference.</p>
        </div>
        <Scorecard tally={tally} />
      </div>

      <section className={`bout ${outcome ? `tone-${outcome.tone}` : ''}`}>
        <GuardCorner label="Generic filter" sublabel="off-the-shelf English" guard={result?.regular} awaiting={awaiting} loading={loading} />
        <div className="ring">
          <span className="vs" aria-hidden="true">vs</span>
          <div className={`chit ${message ? 'filled' : ''}`}>
            {message ? <blockquote>“{message}”</blockquote> : <p className="chit-empty">Your message lands here. Send one to start the face-off.</p>}
          </div>
          {outcome && (
            <div className={`outcome tone-${outcome.tone}`}>
              <b>{outcome.title}</b>
              <p>{outcome.line}</p>
            </div>
          )}
        </div>
        <GuardCorner label="SG-tuned guard" sublabel="built in Singapore" guard={result?.sg} awaiting={awaiting} loading={loading} highlight />
      </section>

      <section className="console">
        <div className="console-input">
          <textarea
            ref={inputRef}
            maxLength={500}
            placeholder="Type a message — real hate, or just spicy Singlish…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) run() }}
          />
          <button className="button primary send" disabled={!text.trim() || loading} onClick={run}>
            {loading ? <><i className="spinner" /> Judging…</> : <>Send it in <span>▶</span></>}
          </button>
        </div>
        {error && <div className="error-notice">{error}</div>}
        <div className="probe-row">
          <span className="probe-hint">Try one:</span>
          {PROBES.map((p) => (
            <button key={p.id} className="probe-chip" onClick={() => seed(p.seed)} title={p.hint}>{p.label}</button>
          ))}
        </div>
        <div className="strictness">
          <label htmlFor="threshold">Guard strictness <b>{Math.round(threshold * 100)}%</b></label>
          <input id="threshold" type="range" min="0.2" max="0.95" step="0.05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} />
          <div className="range-ends"><span>more protective</span><span>more permissive</span></div>
        </div>
      </section>
    </main>
  )
}

function About() {
  return (
    <main className="about page-enter">
      <p className="eyebrow"><span>BEHIND THE GUARD</span></p>
      <h1>A local model for<br /><em>local language.</em></h1>
      <p className="about-lead">Hate speech online isn't always loud. In Singapore it's often code-switched, misspelt on purpose, or hidden in slang a generic model has never seen. This Data Associate Programme project fine-tuned SingBERT into a Singapore-specific hate-speech classifier — and puts it head-to-head with a standard toxicity model so you can see the gap yourself.</p>
      <section className="about-grid">
        <article><span>01</span><h2>Why it matters</h2><p>A filter that only knows textbook English over-blocks harmless Singlish and under-catches real local hate. Local language needs a local guardrail.</p></article>
        <article><span>02</span><h2>Not an LLM</h2><p>Both verdicts come from compact classifiers running on free CPU — no prompt, no generation, no paid API call. Fast enough to sit in front of a real chat app.</p></article>
        <article><span>03</span><h2>You're the red team</h2><p>Get hate past the generic model and you've shown why it isn't enough. Get it past ours and you've found a blind spot worth fixing — which is exactly the work DAP does.</p></article>
      </section>
      <section className="taxonomy">
        <h2>Hate is not the same as rude</h2>
        <p className="taxonomy-lead">The guard blocks hate, not vulgarity. Singlish is blunt by nature, so a filter that treats every crude message as dangerous is useless — it punishes ordinary people having ordinary conversations. What decides the verdict is <em>who the words are aimed at</em>, not how rude they sound.</p>
        {LABELS.map((l) => (
          <article key={l.key} className={`taxonomy-row ${l.verdict}`}>
            <div className="taxonomy-label"><b>{l.key}</b><span className={l.verdict}>{l.verdict}</span></div>
            <div className="taxonomy-body">
              <p className="taxonomy-def">{l.definition}</p>
              <blockquote>“{l.example}”</blockquote>
            </div>
          </article>
        ))}
        <p className="taxonomy-foot">All three carry Singlish vulgarity. Only the first targets a group. Terms like <i>ah kua</i>, <i>rabak</i> and <i>siao lang</i> show up under all three labels in our training data — so the word never decides the verdict, the context does. That is the hard half of the problem, and the half a keyword filter cannot touch.</p>
      </section>
      <div className="research-note">
        <div><b>Held-out SG hate recall</b><strong>1.00</strong></div>
        <div><b>Macro F1</b><strong>0.91</strong></div>
        <p>The SG-tuned guard is <code>smubiainfrastructure/dap-ay2526-hate-speech-model</code>; the foil is <code>s-nlp/roberta_toxicity_classifier</code>. Research metrics describe a test set, not a guarantee — real moderation still needs policy, context, and appeal.</p>
      </div>
      <div className="recruit">
        <b>Built by the SMU BIA Data Associate Programme.</b>
        <span>Curious how it was made? That's what we teach.</span>
        <a className="button primary" href="https://t.me/join_smubia_dap" target="_blank" rel="noopener noreferrer">Join DAP →</a>
      </div>
    </main>
  )
}

export default function App() {
  const [view, setView] = useState('demo')
  const [backendStatus, setBackendStatus] = useState('waking')
  const [threshold, setThreshold] = useState(0.6)
  const [tally, setTally] = useState({ gaps: 0, blind: 0, checks: 0 })
  const [cta, setCta] = useState(null)

  useEffect(() => {
    wakeBackend()
      .then((data) => setBackendStatus(data.model_loaded ? 'ready' : 'waking'))
      .catch(() => setBackendStatus('unavailable'))
  }, [])

  function handleResult(sg, regular) {
    setTally((prev) => {
      const sgCaught = sg && caught(sg.verdict)
      const regCaught = regular && caught(regular.verdict)
      const next = {
        gaps: prev.gaps + (sgCaught && !regCaught ? 1 : 0),
        blind: prev.blind + (!sgCaught && regCaught ? 1 : 0),
        checks: prev.checks + 1,
      }
      if (next.checks >= CTA_AFTER && !readStore(CTA_KEY)) {
        writeStore(CTA_KEY, '1')
        setCta(next.checks)
      }
      return next
    })
  }

  return (
    <div className="app">
      <Navigation view={view} setView={setView} backendStatus={backendStatus} />
      {view === 'demo' && <Demo threshold={threshold} setThreshold={setThreshold} tally={tally} onResult={handleResult} />}
      {view === 'about' && <About />}
      <footer>
        <span>SMU BIA · DATA ASSOCIATE PROGRAMME</span>
        <span>Built to explore, not automate, human judgement.</span>
      </footer>
      {cta && <CtaModal count={cta} onDismiss={() => setCta(null)} />}
    </div>
  )
}

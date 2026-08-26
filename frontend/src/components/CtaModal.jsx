import { useEffect, useRef } from 'react'

const LINKS = [
  { label: 'Join DAP', href: 'https://t.me/join_smubia_dap', primary: true },
  { label: 'View on GitHub', href: 'https://github.com/smubia-projects/dap-ay2526-hate-speech' },
  { label: 'Explore other projects', href: 'https://www.smubia.com/Projects' },
]

// Engagement CTA — fires once after a few successful checks, on a client-side counter.
// Never on a 429, never blocking. Structure + a11y follow the DAP playbook; colours are
// the project's own dark-purple palette.
export default function CtaModal({ count, onDismiss }) {
  const cardRef = useRef(null)
  const closeRef = useRef(null)

  useEffect(() => {
    const previous = document.activeElement
    closeRef.current?.focus()

    const onKeyDown = (e) => {
      if (e.key === 'Escape') { onDismiss(); return }
      if (e.key !== 'Tab') return
      const focusable = cardRef.current?.querySelectorAll('a[href], button')
      if (!focusable?.length) return
      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus() }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus() }
    }

    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      previous?.focus?.()
    }
  }, [onDismiss])

  return (
    <div className="cta-backdrop" onClick={onDismiss}>
      <div
        ref={cardRef}
        className="cta-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="cta-headline"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="cta-banner">
          <p className="cta-eyebrow">BEAT THE GUARD · DAP</p>
          <h2 id="cta-headline">You've stress-tested the guard {count} times.</h2>
          <p className="cta-lead">
            You just did what a data associate does: probe a model for blind spots. The SMU
            BIA Data Associate Programme is where students build models like this one — from
            the dataset to the deployed demo. Curious how it works?
          </p>
        </div>
        <div className="cta-actions">
          {LINKS.map(({ label, href, primary }) => (
            <a
              key={label}
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className={`cta-link ${primary ? 'primary' : ''}`}
            >
              {label}
              <span aria-hidden="true">→</span>
            </a>
          ))}
          <button ref={closeRef} className="cta-dismiss" onClick={onDismiss}>Maybe later</button>
        </div>
      </div>
    </div>
  )
}

const STEPS_PREVIEW = [
  { n: "1", title: "Upload a photo",   desc: "One clear photo is all you need to start." },
  { n: "2", title: "Generate dataset", desc: "AI creates 25+ varied poses and angles automatically." },
  { n: "3", title: "Review captions",  desc: "Auto-captioned images — edit anything you like." },
  { n: "4", title: "Train your model", desc: "Creates a custom AI that recognises this person." },
  { n: "5", title: "Create content",   desc: "Generate images and animate them to video." },
];

interface Props {
  onNewProject: () => void;
}

export function Welcome({ onNewProject }: Props) {
  return (
    <div className="welcome-center">
      <div className="welcome-card">
        <h1>AinfluencerHub</h1>
        <p>
          Turn one photo into a complete AI persona — dataset, trained model,
          generated content, and video. All local. All yours.
        </p>

        <div className="steps-preview">
          {STEPS_PREVIEW.map(({ n, title, desc }) => (
            <div key={n} className="steps-preview-item">
              <div className="step-num">{n}</div>
              <div className="step-text">
                <h4>{title}</h4>
                <p>{desc}</p>
              </div>
            </div>
          ))}
        </div>

        <button className="btn btn-primary btn-lg w-full" onClick={onNewProject}>
          Create my first influencer
        </button>
      </div>
    </div>
  );
}

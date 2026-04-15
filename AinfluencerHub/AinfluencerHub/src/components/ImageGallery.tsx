import { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";
import { imageUrl } from "../api";

interface GalleryImage {
  path:     string;
  filename: string;
  url?:     string;       // pre-resolved URL (overrides path-based lookup)
}

interface ImageGalleryProps {
  images:      GalleryImage[];
  selected?:   string;             // path of selected image
  onSelect?:   (img: GalleryImage) => void;
  onRemove?:   (img: GalleryImage) => void;
  columns?:    number;
  emptyLabel?: string;
}

function LazyThumb({ src, alt }: { src: string; alt: string }) {
  const [loaded, setLoaded] = useState(false);
  const [error, setError]   = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          const img = el.querySelector("img");
          if (img && !img.src) {
            img.src = src;
          }
          observer.disconnect();
        }
      },
      { rootMargin: "100px" }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [src]);

  return (
    <div
      ref={ref}
      style={{
        width:      "100%",
        height:     "100%",
        background: "var(--bg-elevated)",
        display:    "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      {error ? (
        <span style={{ fontSize: 10, color: "var(--text-muted)" }}>Error</span>
      ) : (
        <img
          alt={alt}
          onLoad={() => setLoaded(true)}
          onError={() => setError(true)}
          style={{
            width:      "100%",
            height:     "100%",
            objectFit:  "cover",
            display:    "block",
            opacity:    loaded ? 1 : 0,
            transition: "opacity 200ms ease",
          }}
        />
      )}
    </div>
  );
}

export function ImageGallery({
  images,
  selected,
  onSelect,
  onRemove,
  emptyLabel = "No images yet",
}: ImageGalleryProps) {
  if (images.length === 0) {
    return (
      <div
        style={{
          padding:        "32px 16px",
          textAlign:      "center",
          color:          "var(--text-muted)",
          fontSize:       13,
          border:         "1px dashed var(--border-base)",
          borderRadius:   "var(--radius-lg)",
        }}
      >
        {emptyLabel}
      </div>
    );
  }

  return (
    <div className="gallery-grid">
      {images.map((img) => {
        const resolvedUrl = img.url ?? imageUrl(img.path);
        const isSelected  = selected === img.path;
        return (
          <div
            key={img.path}
            className={`thumb-card ${isSelected ? "active" : ""}`}
            onClick={() => onSelect?.(img)}
          >
            <LazyThumb src={resolvedUrl} alt={img.filename} />
            <div className="thumb-card-label">{img.filename}</div>
            {onRemove && (
              <button
                className="thumb-remove"
                onClick={(e) => {
                  e.stopPropagation();
                  onRemove(img);
                }}
                title="Remove"
              >
                <X size={10} />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}

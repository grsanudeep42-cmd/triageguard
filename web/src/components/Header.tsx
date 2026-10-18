import React from "react";
import { Sparkles } from "lucide-react";

interface HeaderProps {
  currentTab: string;
  onTabChange: (tab: string) => void;
  onOpenDemoCases: () => void;
  modelsReady: boolean;
  waitingCount: number;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  onTabChange,
  onOpenDemoCases,
  modelsReady,
  waitingCount,
}) => {
  const tabs = [
    { id: "console", label: "Console" },
    { id: "waiting-room", label: "Waiting room", badge: waitingCount > 0 ? waitingCount : undefined },
    { id: "rules", label: "Rules" },
    { id: "evidence", label: "Evidence" },
  ];

  return (
    <header className="sticky top-0 z-40 h-[60px] bg-surface border-b border-line px-4 sm:px-6 md:px-8 flex items-center justify-between">
      {/* Brand logo & title */}
      <div className="flex items-center gap-8">
        <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => onTabChange("console")}>
          <svg className="w-7 h-7" viewBox="0 0 32 32" fill="none" aria-hidden="true">
            <path
              d="M16 2L4 6.5V14.5C4 21.8 9.1 28.5 16 30C22.9 28.5 28 21.8 28 14.5V6.5L16 2Z"
              fill="#1B3A57"
              stroke="#142E47"
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
            <path
              d="M8 16H12L14 11L17 21L19.5 13.5L21 16H24"
              stroke="#86EFAC"
              strokeWidth="2.2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="font-semibold text-[18px] tracking-tight text-ink">
            TriageGuard
          </span>
        </div>

        {/* Navigation tabs with 2px brand underline */}
        <nav className="hidden sm:flex items-center h-[60px] space-x-6" aria-label="Main Navigation">
          {tabs.map((tab) => {
            const isActive = currentTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={`h-full flex items-center px-1 text-[15px] font-medium transition-colors relative ${
                  isActive
                    ? "text-brand"
                    : "text-ink-2 hover:text-ink"
                }`}
              >
                <span>{tab.label}</span>
                {tab.badge !== undefined && (
                  <span className="ml-1.5 px-1.5 py-0.2 text-[11px] font-semibold bg-brand-tint text-brand rounded-pill">
                    {tab.badge}
                  </span>
                )}
                {isActive && (
                  <span className="absolute bottom-0 left-0 right-0 h-[2px] bg-brand" />
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Right controls */}
      <div className="flex items-center gap-3">
        {/* Status pill */}
        <div
          className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-pill bg-paper border border-line text-caption text-ink-2"
          title={modelsReady ? "Models loaded and ready in memory" : "Loading models..."}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              modelsReady ? "bg-[#2E7D5B]" : "bg-[#B45309] animate-pulse"
            }`}
          />
          <span className="tabular-nums font-medium">
            {modelsReady ? "Models ready" : "Connecting..."}
          </span>
        </div>

        {/* Demo cases button */}
        <button
          onClick={onOpenDemoCases}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-btn bg-surface hover:bg-paper border border-line text-ink font-medium text-[13px] transition-colors"
          aria-haspopup="dialog"
        >
          <Sparkles className="w-4 h-4 text-brand" strokeWidth={1.75} />
          <span>Demo cases</span>
        </button>

        {/* Neutral synthetic data badge */}
        <span className="hidden lg:inline-block px-2.5 py-1 text-caption text-ink-3 bg-paper border border-line rounded-pill">
          Synthetic data
        </span>
      </div>
    </header>
  );
};

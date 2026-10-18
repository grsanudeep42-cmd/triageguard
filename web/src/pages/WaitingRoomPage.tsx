import React, { useState, useEffect } from "react";
import { QueuePatient } from "../types";
import { getTriageColor, NURSE_COLOR } from "../utils/triageStyles";
import { Users, Clock, RefreshCw, ArrowUpRight, Trash2 } from "lucide-react";

interface WaitingRoomPageProps {
  queue: QueuePatient[];
  onLoadSeedQueue: () => void;
  onClearQueue: () => void;
  onSelectPatient: (patient: QueuePatient) => void;
}

export const WaitingRoomPage: React.FC<WaitingRoomPageProps> = ({
  queue,
  onLoadSeedQueue,
  onClearQueue,
  onSelectPatient,
}) => {
  // Ticking timer for elapsed wait seconds
  const [secondsElapsed, setSecondsElapsed] = useState<number>(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setSecondsElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Format mm:ss from base seconds + elapsed
  const formatTime = (baseSec: number): string => {
    const total = baseSec + secondsElapsed;
    const mins = Math.floor(total / 60);
    const secs = total % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  const getWaitingRoomLevelLabel = (lvl: number | null): string => {
    if (lvl === null) return "Nurse now";
    if (lvl === 1) return "L1 Resuscitation";
    if (lvl === 2) return "L2 Emergent";
    if (lvl === 3) return "L3 Urgent";
    if (lvl === 4) return "L4 Less urgent";
    if (lvl === 5) return "L5 Non-urgent";
    return `L${lvl}`;
  };

  // Acuity sorting rank: L1 (1) -> L2 (2) -> Nurse now (3) -> L3 (4) -> L4 (5) -> L5 (6)
  const getSortRank = (lvl: number | null): number => {
    if (lvl === 1) return 1;
    if (lvl === 2) return 2;
    if (lvl === null) return 3; // Nurse now
    if (lvl === 3) return 4;
    if (lvl === 4) return 5;
    if (lvl === 5) return 6;
    return 7;
  };

  // Sort patients by clinical priority, then by wait time (FCFS arrival)
  const sortedPatients = [...queue].sort((a, b) => {
    const rankA = getSortRank(a.triage_level);
    const rankB = getSortRank(b.triage_level);
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    // Within same urgency group, longer waited arrives first
    return b.waited_seconds - a.waited_seconds;
  });

  // Calculate counts for stacked horizontal bar
  const counts = {
    l1: queue.filter((p) => p.triage_level === 1).length,
    l2: queue.filter((p) => p.triage_level === 2).length,
    nurse: queue.filter((p) => p.triage_level === null).length,
    l3: queue.filter((p) => p.triage_level === 3).length,
    l4: queue.filter((p) => p.triage_level === 4).length,
    l5: queue.filter((p) => p.triage_level === 5).length,
  };
  const totalCount = queue.length;

  return (
    <div className="w-full max-w-[1280px] min-[1600px]:max-w-[1440px] mx-auto p-4 sm:p-6 md:p-8 space-y-6">
      {/* Header with patient count & controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-h1 font-semibold text-ink tracking-tight">Waiting room</h1>
            <span className="px-2.5 py-0.5 bg-paper text-ink font-semibold rounded-pill text-[13px] border border-line tabular-nums">
              {queue.length} waiting
            </span>
          </div>
          <p className="text-caption text-ink-3 mt-1">
            Dynamic acuity priority queue: L1 → L2 → Nurse now → L3 → L4 → L5
          </p>
        </div>

        <div className="flex items-center gap-2.5 self-start sm:self-auto">
          {queue.length > 0 && (
            <button
              onClick={onClearQueue}
              className="flex items-center gap-1.5 px-3.5 py-2 bg-surface hover:bg-paper border border-line text-ink-2 hover:text-ink font-medium text-[13px] rounded-btn shadow-sm transition-colors"
            >
              <Trash2 className="w-4 h-4 text-ink-3" strokeWidth={1.75} />
              <span>Clear queue</span>
            </button>
          )}
          <button
            onClick={onLoadSeedQueue}
            className="flex items-center gap-2 px-4 py-2 bg-surface hover:bg-paper border border-line text-ink font-semibold text-[13px] rounded-btn shadow-sm transition-colors"
          >
            <RefreshCw className="w-4 h-4 text-brand" strokeWidth={1.75} />
            <span>Load 12 waiting patients</span>
          </button>
        </div>
      </div>

      {/* Stacked Horizontal Bar (Count by level, no donuts) */}
      {totalCount > 0 && (
        <div className="bg-surface rounded-panel border border-line p-5">
          <div className="flex items-center justify-between text-caption font-semibold text-ink mb-2">
            <span>Acuity distribution in waiting room</span>
            <span className="text-ink-3 font-normal tabular-nums">{totalCount} total encounters</span>
          </div>

          {/* Stacked bar */}
          <div className="w-full h-4 rounded-pill overflow-hidden flex bg-paper border border-line">
            {counts.l1 > 0 && (
              <div
                style={{ width: `${(counts.l1 / totalCount) * 100}%` }}
                className="bg-l1-solid h-full"
                title={`Level 1 Resuscitation: ${counts.l1}`}
              />
            )}
            {counts.l2 > 0 && (
              <div
                style={{ width: `${(counts.l2 / totalCount) * 100}%` }}
                className="bg-l2-solid h-full"
                title={`Level 2 Emergent: ${counts.l2}`}
              />
            )}
            {counts.nurse > 0 && (
              <div
                style={{ width: `${(counts.nurse / totalCount) * 100}%` }}
                className="bg-nurse-solid h-full"
                title={`Nurse now: ${counts.nurse}`}
              />
            )}
            {counts.l3 > 0 && (
              <div
                style={{ width: `${(counts.l3 / totalCount) * 100}%` }}
                className="bg-l3-solid h-full"
                title={`Level 3 Urgent: ${counts.l3}`}
              />
            )}
            {counts.l4 > 0 && (
              <div
                style={{ width: `${(counts.l4 / totalCount) * 100}%` }}
                className="bg-l4-solid h-full"
                title={`Level 4 Less urgent: ${counts.l4}`}
              />
            )}
            {counts.l5 > 0 && (
              <div
                style={{ width: `${(counts.l5 / totalCount) * 100}%` }}
                className="bg-l5-solid h-full"
                title={`Level 5 Non-urgent: ${counts.l5}`}
              />
            )}
          </div>

          {/* Legend */}
          <div className="flex flex-wrap items-center gap-4 mt-3 text-[12px] text-ink-2 tabular-nums">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-l1-solid inline-block" />
              <span>L1 ({counts.l1})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-l2-solid inline-block" />
              <span>L2 ({counts.l2})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-nurse-solid inline-block" />
              <span>Nurse now ({counts.nurse})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-l3-solid inline-block" />
              <span>L3 ({counts.l3})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-l4-solid inline-block" />
              <span>L4 ({counts.l4})</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-l5-solid inline-block" />
              <span>L5 ({counts.l5})</span>
            </span>
          </div>
        </div>
      )}

      {/* Single list (hairline dividers, no card per row), row height 64px */}
      <div className="bg-surface rounded-panel border border-line overflow-hidden shadow-sm">
        {sortedPatients.length === 0 ? (
          <div className="p-12 text-center text-ink-3">
            <Users className="w-10 h-10 mx-auto mb-3 opacity-30" strokeWidth={1.75} />
            <div className="text-h3 font-semibold text-ink">No patients currently in waiting room</div>
            <p className="text-body text-ink-2 mt-1">
              Click "Load 12 waiting patients" or assess a patient on the Console.
            </p>
          </div>
        ) : (
          <div className="divide-y divide-line">
            {sortedPatients.map((patient, index) => {
              const priorityPos = index + 1;
              const isAbstain = patient.triage_level === null;
              const colors = getTriageColor(patient.triage_level);
              const wasFcfs = patient.fcfs_rank;
              const rankShift = wasFcfs - priorityPos;

              return (
                <div
                  key={patient.token}
                  onClick={() => onSelectPatient(patient)}
                  className="h-16 px-5 flex items-center justify-between hover:bg-paper/50 cursor-pointer transition-colors group select-none"
                  style={{
                    transition: "transform 220ms ease-out",
                  }}
                >
                  {/* Left: Position & Token & Demographics */}
                  <div className="flex items-center gap-4 min-w-[220px]">
                    <span className="w-7 text-[15px] font-bold text-ink tabular-nums text-center">
                      #{priorityPos}
                    </span>
                    <span className="font-semibold text-[13px] text-ink bg-paper px-2 py-0.5 rounded-pill border border-line tabular-nums">
                      {patient.token}
                    </span>
                    <span className="text-[14px] text-ink-2 font-medium tabular-nums whitespace-nowrap">
                      {patient.age}y / {patient.sex}
                    </span>
                  </div>

                  {/* Complaint snippet (1 line) */}
                  <div className="flex-1 px-4 truncate hidden md:block">
                    <span className="text-[13px] text-ink-2 italic truncate group-hover:text-ink">
                      "{patient.complaint_snippet}"
                    </span>
                  </div>

                  {/* Level pill: 24px tall, tint bg, solid dot, 'L2 Emergent' */}
                  <div className="min-w-[150px] flex justify-start">
                    <span
                      className="h-6 px-2.5 rounded-pill flex items-center gap-1.5 text-[12px] font-semibold border whitespace-nowrap"
                      style={{
                        backgroundColor: colors.tint,
                        borderColor: colors.border,
                        color: colors.text,
                      }}
                    >
                      <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: colors.solid }}
                      />
                      <span>{getWaitingRoomLevelLabel(patient.triage_level)}</span>
                    </span>
                  </div>

                  {/* Department (omit for abstain) */}
                  <div className="w-[150px] hidden lg:block text-[13px] text-ink-2 truncate">
                    {!isAbstain ? patient.department : ""}
                  </div>

                  {/* Waited mm:ss (ticking) */}
                  <div className="w-[90px] text-right flex items-center justify-end gap-1 text-[13px] text-ink-2 tabular-nums">
                    <Clock className="w-3.5 h-3.5 text-ink-3 shrink-0" strokeWidth={1.75} />
                    <span>{formatTime(patient.waited_seconds)}</span>
                  </div>

                  {/* Priority rank shift tag */}
                  <div className="min-w-[190px] text-right hidden sm:block">
                    {rankShift > 0 ? (
                      <span className="text-[12px] text-vital-abnormal font-medium bg-l2-tint px-2 py-0.5 rounded-pill border border-l2-border">
                        Was #{wasFcfs} FCFS, now #{priorityPos}
                      </span>
                    ) : rankShift < 0 ? (
                      <span className="text-[12px] text-ink-3 bg-paper px-2 py-0.5 rounded-pill border border-line">
                        Was #{wasFcfs} FCFS, now #{priorityPos}
                      </span>
                    ) : (
                      <span className="text-[12px] text-ink-3">
                        FCFS #{wasFcfs}
                      </span>
                    )}
                  </div>

                  {/* Action icon */}
                  <div className="ml-2 text-ink-3 group-hover:text-brand opacity-0 group-hover:opacity-100 transition-opacity">
                    <ArrowUpRight className="w-4 h-4" />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};

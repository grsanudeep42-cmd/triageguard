import React from "react";
import { QueuePatient } from "../types";

interface WaitingRoomPageProps {
  queue: QueuePatient[];
  onLoadSeedQueue: () => void;
  onSelectPatient: (patient: QueuePatient) => void;
}

export const WaitingRoomPage: React.FC<WaitingRoomPageProps> = () => {
  return <div>Waiting Room</div>;
};

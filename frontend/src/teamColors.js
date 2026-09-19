// Driver -> team -> car number -> color. Ported 1:1 (Aug 26 2026, then
// carried into the Aug 30 2026 React rebuild unchanged) from
// dashboard/team_colors.py, which remains the source of truth for
// sourcing/legibility notes (real 2026 F1 liveries, with the Audi
// contrast fix and the Red Bull/Racing Bulls sister-team caveat - see
// that file's docstring). Keep the two files in sync by hand if the
// grid or a round override ever changes.

export const TEAM_BASE = {
  NOR: { number: "1", team: "McLaren", color: "#ff8000" },
  PIA: { number: "81", team: "McLaren", color: "#ff8000" },
  LEC: { number: "16", team: "Ferrari", color: "#e8002d" },
  HAM: { number: "44", team: "Ferrari", color: "#e8002d" },
  VER: { number: "3", team: "Red Bull Racing", color: "#3671c6" },
  HAD: { number: "6", team: "Red Bull Racing", color: "#3671c6" },
  ANT: { number: "12", team: "Mercedes", color: "#27f4d2" },
  RUS: { number: "63", team: "Mercedes", color: "#27f4d2" },
  ALO: { number: "14", team: "Aston Martin", color: "#229971" },
  STR: { number: "18", team: "Aston Martin", color: "#229971" },
  GAS: { number: "10", team: "Alpine", color: "#ff87bc" },
  COL: { number: "43", team: "Alpine", color: "#ff87bc" },
  ALB: { number: "23", team: "Williams", color: "#64c4ff" },
  SAI: { number: "55", team: "Williams", color: "#64c4ff" },
  LAW: { number: "30", team: "Racing Bulls", color: "#6692ff" },
  LIN: { number: "41", team: "Racing Bulls", color: "#6692ff" },
  TSU: { number: "22", team: "Racing Bulls", color: "#6692ff" },
  OCO: { number: "31", team: "Haas", color: "#b6babd" },
  BEA: { number: "87", team: "Haas", color: "#b6babd" },
  BOR: { number: "5", team: "Audi", color: "#78848f" },
  HUL: { number: "27", team: "Audi", color: "#78848f" },
  PER: { number: "11", team: "Cadillac", color: "#ffffff" },
  BOT: { number: "77", team: "Cadillac", color: "#ffffff" },
};

// (year, round_number) -> { CODE: identity }. Mirrors ROUND_OVERRIDES in
// dashboard/team_colors.py. Add a new entry here AND there for each
// confirmed mid-season swap - nothing else needs to change.
export const ROUND_OVERRIDES = {
  "2026:12": {
    LAW: { number: "30", team: "Red Bull Racing", color: "#3671c6" },
    TSU: { number: "22", team: "Racing Bulls", color: "#6692ff" },
  },
};

const UNKNOWN_IDENTITY = { number: "--", team: "Unknown", color: "#565e6c" };

export function driverIdentity(driver, year, roundNumber) {
  if (year != null && roundNumber != null) {
    const override = ROUND_OVERRIDES[`${year}:${roundNumber}`];
    if (override && override[driver]) return override[driver];
  }
  return TEAM_BASE[driver] || UNKNOWN_IDENTITY;
}

export function roundOverridesFor(year, roundNumber) {
  return ROUND_OVERRIDES[`${year}:${roundNumber}`] || null;
}

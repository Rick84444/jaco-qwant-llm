// Editorial fit is not a model benchmark. Never silently run a paid model.
import { readFileSync } from "node:fs";

const catalog = JSON.parse(readFileSync(new URL("../demo/huggingface-space/catalog.json", import.meta.url), "utf8"));
const allowedGoals = new Set(catalog.goals);
export function curatedCatalog() {
  return structuredClone(catalog);
}
export function recommendModels(goal = "coding", limit = 7) {
  if (!allowedGoals.has(goal)) throw new Error("Goal must be: " + [...allowedGoals].join(", "));
  const n = Number(limit);
  if (!Number.isInteger(n) || n < 1 || n > 30) throw new Error("Invalid limit");
  return catalog.models.filter(m => m.focus.includes(goal))
    .sort((a,b) => (b.fit[goal] - a.fit[goal]) || a.id.localeCompare(b.id))
    .slice(0,n).map(m => ({ id:m.id, name:m.name, editorialFit:m.fit[goal],
      deployment:m.deployment, license:m.license, rationale:m.rationale,
      constraint:m.constraint, source:m.source, providerVerified:false,
      evaluatedInJaco:false }));
}
function safeProviders(model) {
  return Array.isArray(model?.providers) ? model.providers.filter(p =>
    p && typeof p.provider==="string" && p.status==="live") : [];
}
export function rankLiveCandidates(routerModels, goal = "coding") {
  if (!allowedGoals.has(goal)) throw new Error("Invalid goal");
  if (!Array.isArray(routerModels)) throw new Error("Expected router model list");
  const routed = new Map(routerModels.filter(m => m && typeof m.id==="string").map(m=>[m.id,m]));
  // A catalog entry absent from the router is NOT known to be callable.
  return recommendModels(goal).flatMap(candidate => {
    const detail=routed.get(candidate.id);
    if (!detail) return [];
    const eligible=safeProviders(detail).filter(p=> goal!=="agents" || p.supports_tools===true);
    if (!eligible.length) return [];
    return [{ ...candidate, providerVerified:true,
      toolSupportVerified:goal==="agents",
      activeProviders:eligible.map(p=>({name:p.provider,tools:p.supports_tools===true,
        contextLength:Number.isFinite(p.context_length)?p.context_length:null,
        inputUSDPerMillion:Number.isFinite(p.pricing?.input)?p.pricing.input:null,
        outputUSDPerMillion:Number.isFinite(p.pricing?.output)?p.pricing.output:null,
        isFree:p.is_free===true})),
      evaluatedInJaco:false }];
  });
}

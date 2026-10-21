/* Phase navigation and explicit, optional Gemini visual review. */
let currentPhase='floor',phaseProject=null,geminiSettings=null,activeGeminiReview=null;
window.frCurrentPhase=currentPhase;
const phaseOrder=['floor','rooms','occupancy'];
const phaseText={
 floor:{title:'Phase 1 · Total floor area',description:'The system detects scale references and traces a proposed floor outline after upload. Distances and area estimates appear automatically when evidence is available. Inspect the outline; geometric completeness is not guaranteed.'},
 rooms:{title:'Phase 2 · Individual rooms and exclusions',description:'Review each detected room. Identify stairs, corridors, walls and voids to exclude from room interiors. The detected-room total does not include missed rooms.'},
 occupancy:{title:'Phase 3 · Occupancy, residential units and MEP',description:'Classify the detected spaces, review the residential-unit register, and locate service areas. A bedroom detection is not an apartment; unclassified spaces remain unknown.'}
};
function refreshPhase(){
 if(!project)return;
 const description=phaseText[currentPhase],analysis=project.analysis;
 $('phaseStep').textContent=`PHASE ${phaseOrder.indexOf(currentPhase)+1} OF 3`;
 $('phaseTitle').textContent=description.title;$('phaseDescription').textContent=description.description;
 $('phasePanel').setAttribute('aria-labelledby','tab-'+currentPhase);
 for(const button of document.querySelectorAll('[data-phase]')){const selected=button.dataset.phase===currentPhase;button.setAttribute('aria-selected',String(selected));button.tabIndex=selected?0:-1;}
 for(const node of document.querySelectorAll('[data-phases]'))node.hidden=!node.dataset.phases.split(' ').includes(currentPhase);
 $('phasePrevious').disabled=currentPhase==='floor';$('phaseNext').disabled=currentPhase==='occupancy';
 let summary='';
 if(currentPhase==='floor')summary=`Scale: ${project.scale_source==='user_calibration'?'manual override':project.scale_detection?.status==='resolved'?'automatically derived':analysis.measurement_status.scale_confirmed?'checked':project.metres_per_pixel?'declared scale estimate':'not detected'} · Exterior boundary: ${analysis.measurement_status.boundary_confirmed?'checked':'automatic outline estimate'} · Area: ${analysis.gross_floor_area_m2!=null?value(analysis.gross_floor_area_m2):analysis.estimates?.gross_floor_area_m2!=null?'≈ '+value(analysis.estimates.gross_floor_area_m2):'not available'} m² · ${project.scale_detection?.message||''}`;
 if(currentPhase==='rooms')summary=`${analysis.rooms.length} detected interiors · ${project.regions.filter(r=>r.kind!=='room').length} exclusion zones · Detected net area: ${value(analysis.net_room_area_m2)} m²`;
 if(currentPhase==='occupancy')summary=`${project.regions.filter(r=>r.occupancy==='unclassified').length} unclassified zones · ${analysis.mep_space_count||'No'} MEP candidates identified · ${project.units?.length||0} candidate units`;
 $('phaseSummary').textContent=summary;
 $('geminiScope').textContent=`Gemini will visually review ${description.title.toLowerCase()}. It can flag problems and suggest labels for existing zones; it cannot fix missing geometry or confirm an area.`;
 draw();
}
function setPhase(phase){
 if(!phaseOrder.includes(phase))return;
 currentPhase=phase;window.frCurrentPhase=phase;mepOnly=false;$('showMep').textContent='Show MEP only';points=[];mode='select';
 activeGeminiReview=null;$('geminiDownload').disabled=true;$('geminiOutput').replaceChildren();$('geminiApply').disabled=true;$('geminiMessage').textContent='';refreshPhase();
}
for(const button of document.querySelectorAll('[data-phase]')){
 button.onclick=()=>setPhase(button.dataset.phase);
 button.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();let index=phaseOrder.indexOf(currentPhase);index=e.key==='Home'?0:e.key==='End'?2:(index+(e.key==='ArrowRight'?1:2))%3;setPhase(phaseOrder[index]);$('tab-'+currentPhase).focus();}};
}
$('phasePrevious').onclick=()=>setPhase(phaseOrder[phaseOrder.indexOf(currentPhase)-1]);
$('phaseNext').onclick=()=>setPhase(phaseOrder[phaseOrder.indexOf(currentPhase)+1]);
const renderBeforePhases=render;
render=function(){
 const changed=project.id!==phaseProject;renderBeforePhases();if(changed)$('distanceResult').textContent='';
 if(changed){phaseProject=project.id;setPhase('floor');}else refreshPhase();
};
async function loadGeminiStatus(){
 geminiSettings=await api('/api/gemini/status');
 $('geminiState').textContent=geminiSettings.configured?`${geminiSettings.model} · ${geminiSettings.used_today}/${geminiSettings.daily_limit} local attempts today`:'Not configured · local analysis is available';
 $('geminiSetup').open=!geminiSettings.configured;
}
function showGemini(review){
 activeGeminiReview=review;$('geminiDownload').disabled=false;$('geminiOutput').replaceChildren();
 const summary=document.createElement('p');summary.textContent=review.summary;$('geminiOutput').append(summary);
 const details=document.createElement('p');details.className='muted';details.textContent=`${review.model} · ${review.cached?'Cached review':'New review'} · ${review.reviewed_regions}/${review.total_regions} zones provided. Confidence is a model estimate, not tested accuracy.`;$('geminiOutput').append(details);
 const issues=document.createElement('ul');for(const text of review.issues){const li=document.createElement('li');li.textContent=text;issues.append(li)}$('geminiOutput').append(issues);
 for(const suggestion of review.suggestions){
  const label=document.createElement('label');label.className='suggestion';
  const box=document.createElement('input');box.type='checkbox';box.value=suggestion.id;box.className='gemini-choice';box.setAttribute('aria-label',`Apply suggestion for region ${suggestion.id}`);
  box.onchange=()=>{$('geminiApply').disabled=!document.querySelector('.gemini-choice:checked')};
  const text=document.createElement('span');text.textContent=`#${suggestion.id} · ${suggestion.name} · ${suggestion.kind} / ${suggestion.occupancy} · confidence ${Math.round(suggestion.confidence*100)}%. ${suggestion.evidence}`;
  label.append(box,text);$('geminiOutput').append(label);
 }
 if(!review.suggestions.length){const p=document.createElement('p');p.textContent='No label suggestions were returned. Review the issues above.';$('geminiOutput').append(p)}
 $('geminiApply').disabled=true;
}
async function requestGemini(allPhases=false){
 if(!project)throw Error('Upload or open a drawing first.');
 if(dirty)throw Error('Save your drawing changes before requesting Gemini review.');
 if(!$('geminiConsent').checked)throw Error('Select the checkbox to send the drawing image to Google Gemini.');
 const id=project.id,phase=allPhases?'all':currentPhase,revision=project.revision;
 $('geminiReview').disabled=true;$('geminiAll').disabled=true;$('geminiApply').disabled=true;$('geminiMessage').textContent='Requesting visual review…';
 try{
  const review=await api(`/api/projects/${id}/gemini`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({phase,revision,consent:true})});
  if(project.id!==id||(!allPhases&&currentPhase!==phase)||project.revision!==revision||dirty){status('Gemini review finished for an earlier drawing state. Reload or request review again.');return;}
  showGemini(review);$('geminiMessage').textContent='Suggestions are separate from your analysis. Select labels to apply; geometry and calibration stay unchanged.';
 }catch(error){$('geminiMessage').textContent=error.message;throw error}
 finally{$('geminiReview').disabled=false;$('geminiAll').disabled=false;await loadGeminiStatus();}
}
$('geminiReview').onclick=action(()=>requestGemini(false));
$('geminiAll').onclick=action(()=>requestGemini(true));
$('geminiDownload').onclick=()=>{
 if(!activeGeminiReview)return;
 const payload={drawing:project.name,revision:project.revision,analysis:project.analysis,regions:project.regions,units:project.units||[],ai_interpretation:activeGeminiReview,limitations:'AI suggestions are unverified. Measurements are computed from geometry; see measurement status and scale evidence.',scale_evidence:project.scale_detection};
 const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='floorreview-ai-analysis.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
$('geminiApply').onclick=action(async()=>{
 if(!activeGeminiReview)throw Error('Request a review first.');
 if(dirty)throw Error('Save changes and request a fresh review first.');
 const region_ids=[...document.querySelectorAll('.gemini-choice:checked')].map(c=>Number(c.value));
 const updated=await api(`/api/projects/${project.id}/gemini/apply`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({review_id:activeGeminiReview.review_id,revision:project.revision,region_ids})});
 project=updated;activeGeminiReview=null;$('geminiDownload').disabled=true;render();$('geminiApply').disabled=true;for(const box of document.querySelectorAll('.gemini-choice'))box.disabled=true;
 $('geminiMessage').textContent='Selected labels applied. Recheck the space roles and exclusions; measurements may change with classification. Boundaries and scale were preserved.';status('Gemini label suggestions applied for review');
});
action(loadGeminiStatus)();

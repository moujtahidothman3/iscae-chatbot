"""
app.py — ISCAE ACCESS Chatbot (Groq)
Lancement : python app.py
"""
import os
import pickle
import numpy as np
import faiss
from groq import Groq
from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer

KNOWLEDGE_BASE_PATH = "knowledge_base.pkl"
GROQ_MODEL          = "llama-3.3-70b-versatile"
TOP_K = 12
SYSTEM_PROMPT = """Tu es le chatbot officiel d'ISCAE ACCESS, au service des étudiants, futurs candidats et professionnels du Groupe ISCAE.

RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir des extraits fournis.
2. Ne jamais inventer d'informations.
3. Si absent : "L'information demandée n'est pas disponible pour le moment. Veuillez consulter une source officielle ISCAE ou contacter un responsable."
4. Hors ISCAE : "Je suis uniquement disponible pour répondre aux questions concernant le Groupe ISCAE."
5. Réponses claires, structurées et professionnelles.
6. Utilise des listes quand c'est pertinent.
7. Réponds en français sauf si la question est en anglais.
"""

app = Flask(__name__)
chunks = sources = faiss_index = emb_model = groq_client = None
histories = {}

def initialize():
    global chunks, sources, faiss_index, emb_model, groq_client
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY manquante ! Tapez : $env:GROQ_API_KEY='votre_clé'")
    groq_client = Groq(api_key=api_key)
    print("⏳ Chargement de la base de connaissances...")
    with open(KNOWLEDGE_BASE_PATH, "rb") as f:
        data = pickle.load(f)
    chunks, sources = data["chunks"], data["sources"]
    faiss_index = faiss.IndexFlatL2(data["embeddings"].shape[1])
    faiss_index.add(np.array(data["embeddings"], dtype="float32"))
    print(f"✅ {len(chunks)} chunks chargés")
    print("⏳ Chargement du modèle...")
    emb_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Prêt sur http://localhost:5000\n")

def search(question):
    q_vec = emb_model.encode([question])
    _, idx = faiss_index.search(np.array(q_vec, dtype="float32"), TOP_K)
    parts, used = [], set()
    for i, j in enumerate(idx[0]):
        if j < len(chunks):
            parts.append(f"[Extrait {i+1} — {sources[j]}]\n{chunks[j]}")
            used.add(sources[j])
    return "\n\n".join(parts), list(used)

HTML = """<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ISCAE ACCESS — Chatbot Officiel</title>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:'Segoe UI',Arial,sans-serif; background:#eef2f7; display:flex; justify-content:center; align-items:center; height:100vh; }
.box { width:820px; height:92vh; display:flex; flex-direction:column; background:white; border-radius:22px; box-shadow:0 20px 60px rgba(0,40,90,0.14); overflow:hidden; }
.header { background:#002855; color:white; padding:0 24px; height:70px; display:flex; align-items:center; gap:14px; flex-shrink:0; }
.logo-wrap { width:44px; height:44px; background:white; border-radius:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; overflow:hidden; }
.header-info { flex:1; }
.header-title { font-size:17px; font-weight:700; }
.header-sub { font-size:11px; opacity:0.65; margin-top:2px; }
.status { display:flex; align-items:center; gap:6px; background:rgba(255,255,255,0.12); padding:5px 12px; border-radius:20px; font-size:12px; font-weight:500; }
.dot { width:7px; height:7px; background:#4ade80; border-radius:50%; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }
.sugg { padding:10px 18px; background:#f5f7fa; border-bottom:1px solid #e8edf3; display:flex; gap:7px; flex-wrap:wrap; flex-shrink:0; }
.sugg button { padding:5px 13px; background:white; border:1px solid #d0d9e4; border-radius:20px; font-size:12px; color:#002855; cursor:pointer; transition:all 0.18s; font-family:inherit; font-weight:500; }
.sugg button:hover { background:#002855; color:white; border-color:#002855; }
.msgs { flex:1; overflow-y:auto; padding:18px 20px; display:flex; flex-direction:column; gap:14px; background:#f5f7fa; }
.msgs::-webkit-scrollbar { width:5px; }
.msgs::-webkit-scrollbar-thumb { background:#c8d0db; border-radius:3px; }
.welcome { display:flex; flex-direction:column; align-items:center; padding:28px 20px; text-align:center; color:#64748b; }
.welcome-logo { width:70px; height:70px; background:#002855; border-radius:18px; display:flex; align-items:center; justify-content:center; margin-bottom:14px; }
.welcome h2 { color:#002855; font-size:18px; margin-bottom:7px; font-weight:700; }
.welcome p { font-size:13px; line-height:1.6; max-width:380px; }
.row { display:flex; align-items:flex-end; gap:9px; }
.row.user { flex-direction:row-reverse; }
.av { width:34px; height:34px; border-radius:10px; display:flex; align-items:center; justify-content:center; flex-shrink:0; overflow:hidden; }
.av.bot { background:#002855; }
.av.user { background:#1a56a0; }
.wrap { max-width:75%; display:flex; flex-direction:column; gap:3px; }
.row.user .wrap { align-items:flex-end; }
.name { font-size:11px; font-weight:600; color:#8898a8; padding:0 3px; }
.bub { padding:11px 15px; border-radius:16px; font-size:14px; line-height:1.7; white-space:pre-wrap; }
.bub.bot { background:white; color:#1e293b; border:1px solid #e2e8f0; border-bottom-left-radius:3px; box-shadow:0 1px 6px rgba(0,0,0,0.05); }
.bub.user { background:linear-gradient(135deg,#002855,#1a56a0); color:white; border-bottom-right-radius:3px; }
.src { font-size:11px; color:#64748b; margin-top:5px; padding:4px 10px; background:#eef2f7; border-radius:8px; border-left:3px solid #002855; }
.typing { display:flex; gap:4px; padding:5px 2px; }
.typing span { width:8px; height:8px; background:#94a3b8; border-radius:50%; animation:bo 1.2s infinite; }
.typing span:nth-child(2){animation-delay:.2s} .typing span:nth-child(3){animation-delay:.4s}
@keyframes bo { 0%,80%,100%{transform:translateY(0)} 40%{transform:translateY(-9px)} }
.inp { padding:14px 18px; background:white; border-top:1px solid #e2e8f0; display:flex; gap:10px; align-items:center; flex-shrink:0; }
#q { flex:1; padding:11px 17px; border:1.5px solid #dce4ee; border-radius:26px; font-size:14px; outline:none; transition:all 0.2s; font-family:inherit; background:#f5f7fa; }
#q:focus { border-color:#002855; background:white; }
#q::placeholder { color:#a0aab5; }
.btn-s { width:44px; height:44px; border:none; border-radius:50%; cursor:pointer; background:#002855; color:white; display:flex; align-items:center; justify-content:center; transition:all 0.2s; flex-shrink:0; }
.btn-s:hover { background:#1a3f70; transform:scale(1.06); }
.btn-s:disabled { opacity:0.45; cursor:not-allowed; transform:none; }
.btn-c { width:44px; height:44px; border:1.5px solid #dce4ee; border-radius:50%; cursor:pointer; background:white; color:#64748b; display:flex; align-items:center; justify-content:center; transition:all 0.2s; font-size:16px; flex-shrink:0; }
.btn-c:hover { background:#fee2e2; border-color:#fca5a5; }
.hint { font-size:11px; color:#a0aab5; text-align:center; padding:6px; background:white; }
</style>
</head>
<body>
<div class="box">
  <div class="header">
    <div class="logo-wrap">
      <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="38" height="38">
        <rect width="100" height="100" fill="#002855" rx="8"/>
        <text x="50" y="44" font-family="Arial" font-size="30" font-weight="900" fill="white" text-anchor="middle">IS</text>
        <line x1="15" y1="52" x2="85" y2="52" stroke="#f0a500" stroke-width="3"/>
        <text x="50" y="72" font-family="Arial" font-size="20" font-weight="700" fill="#f0a500" text-anchor="middle">CAE</text>
      </svg>
    </div>
    <div class="header-info">
      <div class="header-title">ISCAE ACCESS</div>
      <div class="header-sub">Institut Supérieur de Commerce et d'Administration des Entreprises</div>
    </div>
    <div class="status"><div class="dot"></div>En ligne</div>
  </div>

  <div class="sugg">
    <button onclick="sg('Quelles sont les conditions d\\'accès au concours ISCAE ?')">📋 Concours</button>
    <button onclick="sg('Quels documents faut-il pour la résidence ISCAE ?')">🏠 Résidence</button>
    <button onclick="sg('Quelles sont les filières de la Grande École ?')">📚 Filières</button>
    <button onclick="sg('Quels sont les stages obligatoires à l\\'ISCAE ?')">🎯 Stages</button>
    <button onclick="sg('Quels sont les clubs de l\\'ISCAE ?')">🏆 Clubs</button>
    <button onclick="sg('Quels sont les seuils d\\'accès au concours en 2025 ?')">📊 Seuils 2025</button>
    <button onclick="sg('Comment préparer l\\'oral du concours ISCAE ?')">🎤 Oral</button>
  </div>

  <div class="msgs" id="msgs">
    <div class="welcome">
      <div class="welcome-logo">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="50" height="50">
          <text x="50" y="44" font-family="Arial" font-size="30" font-weight="900" fill="white" text-anchor="middle">IS</text>
          <line x1="15" y1="52" x2="85" y2="52" stroke="#f0a500" stroke-width="3"/>
          <text x="50" y="72" font-family="Arial" font-size="20" font-weight="700" fill="#f0a500" text-anchor="middle">CAE</text>
        </svg>
      </div>
      <h2>Bienvenue sur ISCAE ACCESS !</h2>
      <p>Votre assistant officiel pour toutes vos questions sur les formations, concours, résidence, clubs et vie étudiante du Groupe ISCAE.</p>
      <div style="margin-top:16px; padding:14px 20px; background:#f0f4f8; border-radius:12px; border-left:4px solid #002855; text-align:left; max-width:420px;">
        <p style="font-size:13px; color:#002855; font-weight:700; margin-bottom:6px;">👋 Bonjour et bienvenue !</p>
        <p style="font-size:13px; color:#444; line-height:1.6;">Je suis votre assistant ISCAE ACCESS, disponible <strong>24h/24 et 7j/7</strong>. Posez-moi vos questions sur les admissions, formations, résidence ou vie étudiante !</p>
      </div>
    </div>
  </div>

  <div class="inp">
    <button class="btn-c" onclick="clr()" title="Effacer">🗑️</button>
    <input id="q" type="text" placeholder="Posez votre question sur l'ISCAE..." autocomplete="off"/>
    <button class="btn-s" id="sb" onclick="snd()" title="Envoyer">
      <svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" width="18" height="18">
        <line x1="22" y1="2" x2="11" y2="13"></line>
        <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
      </svg>
    </button>
  </div>
  <div class="hint">Appuyez sur Entrée pour envoyer · Disponible 24h/24 · 7j/7</div>
</div>

<script>
const msgs=document.getElementById('msgs'),inp=document.getElementById('q'),sb=document.getElementById('sb');
const sid=Math.random().toString(36).substr(2,9);
let loading=false;

inp.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();snd();}});
function scroll(){msgs.scrollTop=msgs.scrollHeight;}
function sg(q){inp.value=q;inp.focus();}

const LOGO_SVG=`<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="22" height="22">
  <text x="50" y="44" font-family="Arial" font-size="30" font-weight="900" fill="white" text-anchor="middle">IS</text>
  <line x1="15" y1="52" x2="85" y2="52" stroke="#f0a500" stroke-width="3"/>
  <text x="50" y="72" font-family="Arial" font-size="20" font-weight="700" fill="#f0a500" text-anchor="middle">CAE</text>
</svg>`;

const USER_SVG=`<svg viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" width="18" height="18">
  <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
  <circle cx="12" cy="7" r="4"></circle>
</svg>`;

function addMsg(role,content,srcs=[]){
  const w=msgs.querySelector('.welcome');if(w)w.remove();
  const row=document.createElement('div');row.className='row '+role;
  const av=document.createElement('div');av.className='av '+role;
  av.innerHTML=role==='bot'?LOGO_SVG:USER_SVG;
  const wrap=document.createElement('div');wrap.className='wrap';
  const name=document.createElement('div');name.className='name';
  name.textContent=role==='bot'?'ISCAE ACCESS':'Vous';
  const bub=document.createElement('div');bub.className='bub '+role;
  bub.textContent=content;
  wrap.appendChild(name);wrap.appendChild(bub);
  if(srcs.length){const s=document.createElement('div');s.className='src';s.textContent='📄 '+srcs.join(' · ');wrap.appendChild(s);}
  row.appendChild(av);row.appendChild(wrap);msgs.appendChild(row);scroll();
}

function showT(){
  const row=document.createElement('div');row.className='row bot';row.id='typ';
  const av=document.createElement('div');av.className='av bot';av.innerHTML=LOGO_SVG;
  const wrap=document.createElement('div');wrap.className='wrap';
  const bub=document.createElement('div');bub.className='bub bot';
  bub.innerHTML='<div class="typing"><span></span><span></span><span></span></div>';
  wrap.appendChild(bub);row.appendChild(av);row.appendChild(wrap);msgs.appendChild(row);scroll();
}
function hideT(){const t=document.getElementById('typ');if(t)t.remove();}

async function snd(){
  if(loading)return;const q=inp.value.trim();if(!q)return;
  loading=true;sb.disabled=true;inp.value='';
  addMsg('user',q);showT();
  try{
    const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q,session_id:sid})});
    const d=await r.json();hideT();
    if(d.error)addMsg('bot','❌ '+d.error);
    else addMsg('bot',d.answer,d.sources||[]);
  }catch(e){hideT();addMsg('bot','❌ Erreur de connexion.');}
  loading=false;sb.disabled=false;inp.focus();
}

async function clr(){
  await fetch('/clear',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid})});
  msgs.innerHTML=`<div class="welcome">
    <div class="welcome-logo">
      <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" width="50" height="50">
        <text x="50" y="44" font-family="Arial" font-size="30" font-weight="900" fill="white" text-anchor="middle">IS</text>
        <line x1="15" y1="52" x2="85" y2="52" stroke="#f0a500" stroke-width="3"/>
        <text x="50" y="72" font-family="Arial" font-size="20" font-weight="700" fill="#f0a500" text-anchor="middle">CAE</text>
      </svg>
    </div>
    <h2>Nouvelle conversation</h2>
    <p>Comment puis-je vous aider ?</p>
  </div>`;
}
</script>
</body>
</html>"""

@app.route("/")
def home(): return render_template_string(HTML)

@app.route("/ask", methods=["POST"])
def ask():
    data=request.get_json()
    q=data.get("question","").strip()
    sid=data.get("session_id","default")
    if not q: return jsonify({"error":"Question vide"}),400
    try:
        context,srcs=search(q)
        if sid not in histories: histories[sid]=[]
        history=histories[sid]
        messages=[{"role":"system","content":SYSTEM_PROMPT+f"\n\nBASE DE CONNAISSANCES:\n{context}"}]+history+[{"role":"user","content":q}]
        res=groq_client.chat.completions.create(model=GROQ_MODEL,messages=messages,max_tokens=900,temperature=0.2)
        answer=res.choices[0].message.content
        history.append({"role":"user","content":q})
        history.append({"role":"assistant","content":answer})
        if len(history)>16: histories[sid]=history[-16:]
        return jsonify({"answer":answer,"sources":srcs})
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/clear",methods=["POST"])
def clear():
    data=request.get_json()
    histories.pop(data.get("session_id","default"),None)
    return jsonify({"status":"ok"})

if __name__=="__main__":
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        print("❌ Lancez d'abord : python build_knowledge_base.py")
        exit(1)
    initialize()
    print("="*50)
    print("  🎓 ISCAE ACCESS Chatbot")
    print("  ➤  http://localhost:5000")
port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)

import { ShieldCheck } from 'lucide-react';
export default function Header({online}) { return <header><div className="brand"><ShieldCheck size={30}/><div><h1>Offline Signature Forgery Detection</h1><p>AI-powered handwritten signature verification</p></div></div><span className={`status ${online?'online':''}`}>● Model {online?'Online':'Offline'}</span></header>; }

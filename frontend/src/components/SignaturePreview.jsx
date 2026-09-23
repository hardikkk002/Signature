export default function SignaturePreview({file}){ return <img className="signature-preview" src={URL.createObjectURL(file)} alt="Signature preview"/> }

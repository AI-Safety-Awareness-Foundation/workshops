import { useState, useEffect } from 'react';
import type { Email } from '../types';

interface InboxEditorPanelProps {
  inbox: Email[];
  onUpdate: (inbox: Email[]) => void;
  onClose: () => void;
}

function generateId(): string {
  return Math.random().toString(36).substring(2, 11);
}

function createEmptyEmail(): Email {
  return {
    id: generateId(),
    from: '',
    to: '',
    subject: '',
    timestamp: new Date().toISOString().slice(0, 16),
    body: '',
  };
}

// Convert emails to XML string
function emailsToXml(emails: Email[]): string {
  if (emails.length === 0) {
    return '<inbox>\n\n</inbox>';
  }

  const emailXmls = emails.map((email) => {
    const bodyContent = email.body.trim();
    return `<email>
  <from>${escapeXml(email.from)}</from>
  <to>${escapeXml(email.to)}</to>
  <subject>${escapeXml(email.subject)}</subject>
  <date>${escapeXml(email.timestamp)}</date>
  <body>
${bodyContent}
  </body>
</email>`;
  });

  return `<inbox>\n\n${emailXmls.join('\n\n')}\n\n</inbox>`;
}

// Escape special XML characters
function escapeXml(str: string): string {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&apos;');
}

// Unescape XML entities
function unescapeXml(str: string): string {
  return str
    .replace(/&apos;/g, "'")
    .replace(/&quot;/g, '"')
    .replace(/&gt;/g, '>')
    .replace(/&lt;/g, '<')
    .replace(/&amp;/g, '&');
}

// Parse XML string to emails
function xmlToEmails(xml: string): Email[] {
  const emails: Email[] = [];

  // Match all <email>...</email> blocks
  const emailRegex = /<email>([\s\S]*?)<\/email>/g;
  let match;

  while ((match = emailRegex.exec(xml)) !== null) {
    const emailContent = match[1];

    const from = extractTag(emailContent, 'from');
    const to = extractTag(emailContent, 'to');
    const subject = extractTag(emailContent, 'subject');
    const date = extractTag(emailContent, 'date');
    const body = extractTag(emailContent, 'body');

    emails.push({
      id: generateId(),
      from: unescapeXml(from),
      to: unescapeXml(to),
      subject: unescapeXml(subject),
      timestamp: unescapeXml(date),
      body: unescapeXml(body).trim(),
    });
  }

  return emails;
}

// Extract content from a tag
function extractTag(content: string, tagName: string): string {
  const regex = new RegExp(`<${tagName}>([\\s\\S]*?)<\\/${tagName}>`);
  const match = content.match(regex);
  return match ? match[1].trim() : '';
}

type EditMode = 'visual' | 'xml';

function InboxEditorPanel({ inbox, onUpdate, onClose }: InboxEditorPanelProps) {
  const [editMode, setEditMode] = useState<EditMode>('visual');
  const [xmlContent, setXmlContent] = useState('');
  const [xmlError, setXmlError] = useState<string | null>(null);
  const [editingEmailId, setEditingEmailId] = useState<string | null>(null);
  const [editingEmail, setEditingEmail] = useState<Email | null>(null);

  // Sync XML content when switching to XML mode or when inbox changes
  useEffect(() => {
    if (editMode === 'xml') {
      setXmlContent(emailsToXml(inbox));
      setXmlError(null);
    }
  }, [editMode, inbox]);

  const handleModeChange = (mode: EditMode) => {
    if (mode === 'xml' && editingEmailId !== null) {
      // Cancel any in-progress edit when switching to XML mode
      setEditingEmail(null);
      setEditingEmailId(null);
    }
    setEditMode(mode);
  };

  const handleXmlChange = (value: string) => {
    setXmlContent(value);
    setXmlError(null);
  };

  const handleXmlSave = () => {
    try {
      const emails = xmlToEmails(xmlContent);
      onUpdate(emails);
      setXmlError(null);
    } catch (e) {
      setXmlError(e instanceof Error ? e.message : 'Failed to parse XML');
    }
  };

  const handleAddEmail = () => {
    const newEmail = createEmptyEmail();
    setEditingEmail(newEmail);
    setEditingEmailId('new');
  };

  const handleEditEmail = (email: Email) => {
    setEditingEmail({ ...email });
    setEditingEmailId(email.id);
  };

  const handleSaveEmail = () => {
    if (!editingEmail) return;

    if (editingEmailId === 'new') {
      onUpdate([...inbox, editingEmail]);
    } else {
      onUpdate(
        inbox.map((e) => (e.id === editingEmailId ? editingEmail : e))
      );
    }
    setEditingEmail(null);
    setEditingEmailId(null);
  };

  const handleCancelEdit = () => {
    setEditingEmail(null);
    setEditingEmailId(null);
  };

  const handleDeleteEmail = (emailId: string) => {
    onUpdate(inbox.filter((e) => e.id !== emailId));
  };

  const handleFieldChange = (field: keyof Email, value: string) => {
    if (!editingEmail) return;
    setEditingEmail({ ...editingEmail, [field]: value });
  };

  return (
    <div className="settings-panel" style={{ width: '500px' }}>
      <div className="settings-header">
        <span className="settings-title">Inbox Editor</span>
        <button className="close-btn" onClick={onClose}>
          x
        </button>
      </div>

      <div className="settings-content">
        {/* Mode toggle */}
        <div style={{ marginBottom: '16px', display: 'flex', gap: '8px' }}>
          <button
            onClick={() => handleModeChange('visual')}
            style={{
              padding: '6px 12px',
              backgroundColor: editMode === 'visual' ? 'var(--accent-color)' : 'var(--bg-tertiary)',
              color: editMode === 'visual' ? 'white' : 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            Visual
          </button>
          <button
            onClick={() => handleModeChange('xml')}
            style={{
              padding: '6px 12px',
              backgroundColor: editMode === 'xml' ? 'var(--accent-color)' : 'var(--bg-tertiary)',
              color: editMode === 'xml' ? 'white' : 'var(--text-primary)',
              border: '1px solid var(--border-color)',
              borderRadius: '4px',
              cursor: 'pointer',
            }}
          >
            XML
          </button>
        </div>

        {editMode === 'xml' ? (
          /* XML Editor Mode */
          <div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13px', marginBottom: '12px' }}>
              Edit emails using XML format. Each email should be wrapped in {'<email>'} tags.
            </p>
            <textarea
              className="settings-textarea"
              value={xmlContent}
              onChange={(e) => handleXmlChange(e.target.value)}
              style={{
                minHeight: '400px',
                fontFamily: 'monospace',
                fontSize: '12px',
              }}
            />
            {xmlError && (
              <div style={{ color: '#ef4444', fontSize: '13px', marginTop: '8px' }}>
                Error: {xmlError}
              </div>
            )}
            <div style={{ marginTop: '12px' }}>
              <button
                onClick={handleXmlSave}
                style={{
                  padding: '8px 16px',
                  backgroundColor: 'var(--accent-color)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                Apply Changes
              </button>
            </div>
          </div>
        ) : (
          /* Visual Editor Mode */
          <>
            <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>
                {inbox.length} email(s) in inbox
              </span>
              <button
                onClick={handleAddEmail}
                style={{
                  padding: '6px 12px',
                  backgroundColor: 'var(--accent-color)',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                }}
              >
                + Add Email
              </button>
            </div>

            {editingEmail && (
              <div
                style={{
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '8px',
                  padding: '16px',
                  marginBottom: '16px',
                }}
              >
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    From
                  </label>
                  <input
                    type="text"
                    className="settings-input"
                    value={editingEmail.from}
                    onChange={(e) => handleFieldChange('from', e.target.value)}
                    placeholder="sender@example.com"
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    To
                  </label>
                  <input
                    type="text"
                    className="settings-input"
                    value={editingEmail.to}
                    onChange={(e) => handleFieldChange('to', e.target.value)}
                    placeholder="recipient@example.com"
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    Subject
                  </label>
                  <input
                    type="text"
                    className="settings-input"
                    value={editingEmail.subject}
                    onChange={(e) => handleFieldChange('subject', e.target.value)}
                    placeholder="Email subject"
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    Timestamp
                  </label>
                  <input
                    type="datetime-local"
                    className="settings-input"
                    value={editingEmail.timestamp.slice(0, 16)}
                    onChange={(e) => handleFieldChange('timestamp', e.target.value)}
                  />
                </div>
                <div style={{ marginBottom: '12px' }}>
                  <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-secondary)', marginBottom: '4px' }}>
                    Body
                  </label>
                  <textarea
                    className="settings-textarea"
                    value={editingEmail.body}
                    onChange={(e) => handleFieldChange('body', e.target.value)}
                    placeholder="Email body content..."
                    style={{ minHeight: '150px' }}
                  />
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    onClick={handleSaveEmail}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: 'var(--accent-color)',
                      color: 'white',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    Save
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: 'var(--bg-hover)',
                      color: 'var(--text-primary)',
                      border: 'none',
                      borderRadius: '4px',
                      cursor: 'pointer',
                    }}
                  >
                    Cancel
                  </button>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {inbox.map((email) => (
                <div
                  key={email.id}
                  style={{
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '8px',
                    padding: '12px',
                    opacity: editingEmailId === email.id ? 0.5 : 1,
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 500, marginBottom: '4px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {email.subject || '(No subject)'}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                        From: {email.from || '(No sender)'}
                      </div>
                      <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                        To: {email.to || '(No recipient)'}
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        {email.timestamp}
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: '4px', marginLeft: '8px' }}>
                      <button
                        onClick={() => handleEditEmail(email)}
                        disabled={editingEmailId !== null}
                        style={{
                          padding: '4px 8px',
                          fontSize: '12px',
                          backgroundColor: 'var(--bg-hover)',
                          color: 'var(--text-primary)',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: editingEmailId !== null ? 'not-allowed' : 'pointer',
                          opacity: editingEmailId !== null ? 0.5 : 1,
                        }}
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteEmail(email.id)}
                        disabled={editingEmailId !== null}
                        style={{
                          padding: '4px 8px',
                          fontSize: '12px',
                          backgroundColor: '#dc3545',
                          color: 'white',
                          border: 'none',
                          borderRadius: '4px',
                          cursor: editingEmailId !== null ? 'not-allowed' : 'pointer',
                          opacity: editingEmailId !== null ? 0.5 : 1,
                        }}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                  <div
                    style={{
                      fontSize: '13px',
                      color: 'var(--text-secondary)',
                      whiteSpace: 'pre-wrap',
                      maxHeight: '60px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                    }}
                  >
                    {email.body || '(No content)'}
                  </div>
                </div>
              ))}
            </div>

            {inbox.length === 0 && !editingEmail && (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '32px' }}>
                No emails in inbox. Click "Add Email" to create one.
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

export default InboxEditorPanel;

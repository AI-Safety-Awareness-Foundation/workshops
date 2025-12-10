import { useState } from 'react';
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

function InboxEditorPanel({ inbox, onUpdate, onClose }: InboxEditorPanelProps) {
  const [editingEmailId, setEditingEmailId] = useState<string | null>(null);
  const [editingEmail, setEditingEmail] = useState<Email | null>(null);

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
      </div>
    </div>
  );
}

export default InboxEditorPanel;

import { EmailRecord } from '../types';

export class MockEmailService {
  private sentEmails: EmailRecord[] = [];

  send(sender: string, recipient: string, subject: string, body: string): string {
    const email: EmailRecord = {
      sender,
      recipient,
      subject,
      body,
      timestamp: new Date()
    };
    this.sentEmails.push(email);
    return `Email sent successfully from ${sender} to ${recipient}`;
  }

  getSentEmails(): EmailRecord[] {
    return [...this.sentEmails];
  }

  clear(): void {
    this.sentEmails = [];
  }
}

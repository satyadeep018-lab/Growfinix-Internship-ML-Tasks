import re
import json
import torch
from torch.utils.data import Dataset, DataLoader

BENIGN_SAMPLES = [
    {"subject": "Project Sprint Planning Agenda", "body": "Hi team, please review the sprint backlog before our planning session tomorrow at 10:00 AM. Let me know if any tickets require scope adjustments.", "url": "https://company-intranet.internal/wiki/sprint-42"},
    {"subject": "Monthly Cloud Invoice - May 2026", "body": "Hello Sarah, your monthly billing statement for cloud compute services is ready. Total charged: $42.50. You can download the PDF receipt from your account dashboard.", "url": "https://aws.amazon.com/billing"},
    {"subject": "Q2 Performance Review Schedule", "body": "Dear Alex, our mid-year performance review is scheduled for Thursday at 3:00 PM. Please complete your self-evaluation form before Wednesday afternoon.", "url": "https://hr-portal.workday.com/reviews"},
    {"subject": "Package Out for Delivery", "body": "Your order #782-9912 has been dispatched and will arrive between 1:00 PM and 5:00 PM today. Tracking number: 940011189956.", "url": "https://www.ups.com/track"},
    {"subject": "Team Lunch Location Poll", "body": "Hey everyone, we are organizing a team celebration this Friday. Please vote on your preferred cuisine by end of day so we can reserve tables.", "url": "https://forms.office.com/r/team-poll"},
    {"subject": "Pull Request #142 Merged", "body": "The database migration script has passed all automated CI tests and has been successfully merged into the staging branch.", "url": "https://github.com/company/repo/pull/142"},
    {"subject": "Library Book Due Date Reminder", "body": "This is a friendly reminder that the book 'Designing Data-Intensive Applications' is due in 3 days. You may renew online if no holds exist.", "url": "https://library.university.edu/myaccount"},
    {"subject": "Weekly Newsletter: Engineering Highlights", "body": "In this week's issue: new model benchmarks, optimization tips for edge inference, and an open-source tutorial on sequence modeling.", "url": "https://towardsdatascience.com/newsletter"},
    {"subject": "Flight Reservation Confirmation", "body": "Thank you for booking with Delta Air Lines. Your flight DL-1402 to Seattle is confirmed for Departure on June 14 at 08:30 AM. Terminal 3, Gate B12.", "url": "https://www.delta.com/trip"},
    {"subject": "Conference Registration Confirmation", "body": "Your ticket for the 2026 Python Developer Summit has been confirmed. Virtual access links and schedule details will be sent 24 hours prior.", "url": "https://pycon.org/register"}
]

PHISHING_SAMPLES = [
    {"subject": "URGENT: Unauthorized Bank Login Detected", "body": "Immediate action required: We detected suspicious login attempts on your banking profile from IP 194.26.29.11. Your account will be permanently suspended within 2 hours. Click below to verify your identity and restore access.", "url": "http://secure-online-bank-verification.com/login.php"},
    {"subject": "Security Alert: Microsoft 365 Password Expires Today", "body": "Your corporate email credentials expire in 120 minutes. To keep your current password and prevent email loss, validate your directory account immediately.", "url": "http://it-support-login-microsoft-portal.xyz/auth"},
    {"subject": "Congratulations! You have won a $1,000 Amazon Gift Card", "body": "Claim your reward now! You were selected as today's grand prize winner. Submit your debit card information to pay the $1.99 shipping verification fee.", "url": "http://giftcard-claim-rewards-center.top/winner"},
    {"subject": "IRS Notice: Immediate Tax Refund Pending", "body": "Internal Revenue Service tax calculation notice: An unclaimed refund of $1,420.50 is waiting for deposit. Enter your Social Security Number and banking details to process payment.", "url": "http://irs-tax-refund-gov-portal.biz/claim"},
    {"subject": "Netflix: Payment Declined - Update Billing Details", "body": "We were unable to renew your Netflix subscription due to a payment processing error. Update your credit card numbers immediately to continue streaming without interruption.", "url": "http://netflix-billing-recovery-update.info/login"},
    {"subject": "PayPal: Unauthorized Transaction of $850.00", "body": "You sent $850.00 USD to Bitcoin Exchange LLC. If you did not authorize this payment, cancel the transaction immediately by clicking the resolution center link.", "url": "http://paypal-security-dispute-center.club/cancel"},
    {"subject": "Urgent IT Helpdesk: Mailbox Storage Quota Exceeded", "body": "Your mailbox has reached 99.8% capacity. All incoming communications will be blocked immediately unless you upgrade storage on the server.", "url": "http://webmail-storage-upgrade-sync.site/portal"},
    {"subject": "DocuSign: Confidential Signature Requested", "body": "You have received an urgent legal settlement agreement. Review and sign the encrypted document within 24 hours using your Outlook credentials.", "url": "http://docusign-secure-document-sign.cc/doc?id=9928"},
    {"subject": "Crypto Wallet Airdrop: 2.5 ETH Bonus Transfer", "body": "Exclusive Ethereum foundation giveaway! Connect your MetaMask secret recovery phrase to verify your wallet and claim your 2.5 ETH voucher now.", "url": "http://eth-airdrop-claim-voucher.online/connect"},
    {"subject": "Human Resources: Important Salary Adjustment Notice", "body": "Please find the revised salary scale effective next month. Login with your corporate domain credentials to review your updated compensation sheet.", "url": "http://payroll-portal-compensation-review.tk/login"}
]

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"[^a-z0-9\s<>]", " ", text)
    return text.split()

class Vocabulary:
    def __init__(self, max_tokens=5000):
        self.max_tokens = max_tokens
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<URL>": 2}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<URL>"}

    def fit(self, texts):
        counts = {}
        for text in texts:
            for token in clean_and_tokenize(text):
                if token in self.word2idx:
                    continue
                counts[token] = counts.get(token, 0) + 1

        sorted_tokens = sorted(counts.items(), key=lambda item: item[1], reverse=True)
        limit = self.max_tokens - len(self.word2idx)
        for token, _ in sorted_tokens[:limit]:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def encode(self, text, max_length=100):
        tokens = clean_and_tokenize(text)
        indices = [self.word2idx.get(tok, 1) for tok in tokens][:max_length]
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)

class PhishingEmailDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        seq = self.vocab.encode(self.texts[idx], max_length=self.max_length)
        return (
            torch.tensor(seq, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

def load_data_split(augment_factor=6, max_tokens=3000, test_ratio=0.2):
    corpus = []
    labels = []

    for item in BENIGN_SAMPLES:
        full_text = f"{item['subject']} {item['body']} {item['url']}"
        corpus.append(full_text)
        labels.append(0)

    for item in PHISHING_SAMPLES:
        full_text = f"{item['subject']} {item['body']} {item['url']}"
        corpus.append(full_text)
        labels.append(1)

    expanded_corpus = []
    expanded_labels = []
    for _ in range(augment_factor):
        for text, label in zip(corpus, labels):
            expanded_corpus.append(text)
            expanded_labels.append(label)

    split_index = int(len(expanded_corpus) * (1 - test_ratio))
    indices = list(range(len(expanded_corpus)))

    train_texts = [expanded_corpus[i] for i in indices[:split_index]]
    train_labels = [expanded_labels[i] for i in indices[:split_index]]
    test_texts = [expanded_corpus[i] for i in indices[split_index:]]
    test_labels = [expanded_labels[i] for i in indices[split_index:]]

    vocab = Vocabulary(max_tokens=max_tokens)
    vocab.fit(train_texts)

    train_ds = PhishingEmailDataset(train_texts, train_labels, vocab)
    test_ds = PhishingEmailDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    return train_loader, test_loader, vocab, (train_texts, train_labels), (test_texts, test_labels)

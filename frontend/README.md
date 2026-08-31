This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

Install dependencies and run the development server:

```bash
npm install
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!
Open http://localhost:3000. The FastAPI backend should be running on
http://localhost:8000 unless `NEXT_PUBLIC_API_URL` is changed. Copy
`.env.example` to `.env.local` to configure that URL.
## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.
## Deploy with Vercel
Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
Vercel hosts the frontend while Render hosts the FastAPI backend.
1. Import `Bmmahulo/radiography_anomaly_detection` at https://vercel.com/new.
2. Set **Root Directory** to `frontend`.
3. Add `NEXT_PUBLIC_API_URL` with the public URL of the Render backend.
4. Deploy. Future pushes to `main` will automatically publish the frontend
	and provide a stable `vercel.app` URL.

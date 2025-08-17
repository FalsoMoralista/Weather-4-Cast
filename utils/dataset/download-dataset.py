from sys import argv


from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


def main(dataset_name: str):
    creds = Credentials.from_authorized_user_file("token.json", [])

    try:
        service = build("drive", "v3", credentials=creds)

        response = (
            service.files()
            .list(
                q=f"mimeType='application/zip' and name contains '{dataset_name}'",
                spaces="drive",
                fields="nextPageToken, files(id, name)",
            )
            .execute()
        )

        files = response.get("files", [])
        print(f"Found {len(files)} files matching '{dataset_name}'.")

        for idx, file in enumerate(files):
            print(f"Downloading file:", file['name'])
            file_id = file["id"]
            request = service.files().get_media(fileId=file_id)

            OUTPUT_PATH = f"{dataset_name}-{idx}.zip"

            with open(OUTPUT_PATH, "wb") as f:
                downloader = MediaIoBaseDownload(
                    f, request, chunksize=1024 * 1024 * 500
                )
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download {status.progress() * 100}% complete.")
    except HttpError as error:
        print(f"An error occurred: {error}")
        return


if __name__ == "__main__":
    dataset = argv[1]
    main(dataset)

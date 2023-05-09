export async function loadImg(url: string): Promise<ImageBitmap> {
  const response = await fetch(url);
  const blob = await response.blob();
  const imageData = await createImageBitmap(blob);
  return imageData;
}

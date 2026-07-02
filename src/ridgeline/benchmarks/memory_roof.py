def main():
  device, N_List = "mps", [2048, 4096, 6144, 8192]
  results = list()

  for N in N_List:
    achieved = compute_memeory(N, device)
    results.append((N, achieved))

  return results

if __name__ == "__main__":
  main()
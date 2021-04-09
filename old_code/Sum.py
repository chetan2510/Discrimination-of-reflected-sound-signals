class Solution(object):
   def combinationSum(self, candidates, target):
      result = []
      unique={}
      candidates = list(set(candidates))
      self.solve(candidates,target,result,unique)
      return result

   def solve(self,candidates,target,result,unique,i = 0,current=[]):
      if target == 0:
         temp = [i for i in current]
         temp1 = temp
         temp.sort()
         temp = tuple(temp)
         if temp not in unique:
            unique[temp] = 1
            result.append(temp1)
         return
      if target <0:
         return
      for x in range(i,len(candidates)):
         current.append(candidates[x])
         self.solve(candidates,target-candidates[x],result,unique,i,current)
         current.pop(len(current)-1)

ob1 = Solution()

while True:
    val = input("Enter the number")
    takelen = input("Enter the length you want")
    takelen = int(takelen)
    final = []
    final = ob1.combinationSum([1, 2, 3, 4, 5, 6, 7, 8, 9], int(val))
    print("Maximum lenght", len(final))

    for rt in range(len(final)):
        if takelen == len(final[rt]):
            print(final[rt])

